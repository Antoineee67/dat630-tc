/****************************************************************************/
/* TREEGRAV.C: routines to compute gravity. Public routines: gravcalc().    */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include <assert.h>
#include <cudagravsum.cuh>
#include <stdint.h>
#include "komihash.h"

#include "stdinc.h"
#include "mathfns.h"
#include "vectmath.h"
#include "treedefs.h"
#include "treecode.h"
#include "mpi.h"
#include <omp.h>

#define OMP_DEPTH_THRESHOLD 6

/* Local routines to perform force calculations. */

local bool accept(nodeptr, real, vector);

local void walktree(nodeptr *active_list, uint32_t active_list_len,
                    nodeptr current_node, real current_node_size, vector current_node_midpoint,
                    cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail, uint32_t depth);

local void gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail);

/*
 * GRAVCALC: perform force calculation on all particles.
 */

void gravcalc(void) {
    //body_list = malloc(size * sizeof(node));
    double cpustart;
    vector rmid;
    cpustart = cputime(); /* record time, less alloc  */
    actmax = nbbcalc = nbccalc = 0; /* zero cumulative counters */
    CLRV(rmid); /* set center of root cell  */
    walktree((nodeptr *) &root, 1, (nodeptr) root, rsize, rmid, NULL, NULL, 0); /* scan tree, update forces */
    cpuforce = cputime() - cpustart; /* store CPU time w/o alloc */
}

static void walksubtree(nodeptr root_node, nodeptr *active_list, uint32_t active_list_len, real current_node_size,
                        const real *current_node_midpoint, cell_ll_entry_t *cell_list_tail,
                        cell_ll_entry_t *body_list_tail, uint32_t depth, real poff) {
    if (depth == mpi_depth && mpi_rank != komihash(root_node->pos, sizeof(root_node->pos), 0) % mpi_numproc) {
        // skip this subtree on this mpi process
        return;
    }

    vector nmid;
    for (int k = 0; k < NDIM; k++) /* locate each's midpoint   */
        nmid[k] = current_node_midpoint[k] + (Pos(root_node)[k] < current_node_midpoint[k] ? -poff : poff);
    walktree(active_list, active_list_len, root_node, current_node_size / 2, nmid, cell_list_tail, body_list_tail,
             depth);
}

local void walksub(nodeptr *active_list, uint32_t active_list_len,
                   nodeptr current_node, real current_node_size, const vector current_node_midpoint,
                   cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail, uint32_t depth) {
    depth++;
    real poff = current_node_size / 4; /* precompute mid. offset   */

    if (Type(current_node) == CELL) {
        /* fanout over descendents  */
        nodeptr points[10];
        int size = 0;

        for (nodeptr q = More(current_node); q != Next(current_node); q = Next(q)) {
            points[size] = q;
            size++;
            assert(size < sizeof(points) / sizeof(points[0]));
        }


        if (depth < omp_threshold) {
            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int nt = omp_get_num_threads();
                if (nt>1){
                    printf("Number of threads (inside parallel): %d\n", nt);                    
                }

                walksubtree(points[i], active_list, active_list_len, current_node_size, current_node_midpoint,
                            cell_list_tail, body_list_tail,
                            depth, poff);
            }
        } else {
            for (int i = 0; i < size; i++) {
                walksubtree(points[i], active_list, active_list_len, current_node_size, current_node_midpoint,
                            cell_list_tail, body_list_tail,
                            depth, poff);
            }
        }
    } else {
        walksubtree(current_node, active_list, active_list_len, current_node_size, current_node_midpoint,
                    cell_list_tail, body_list_tail,
                    depth, poff);
    }
}


local void walktree(nodeptr *active_list, uint32_t active_list_len,
                    nodeptr current_node, real current_node_size, vector current_node_midpoint,
                    cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail, uint32_t depth) {
    if (!Update(current_node)) {
        return;
    }

    //TODO: Better way to do this?
    cell_ll_entry_t free_enties[active_list_len];
    uint32_t free_entry_index = 0;

    const uint32_t max_new_active_list_len = active_list_len * 8;
    //Worst case scenario where each active node has 8 subnodes
    nodeptr next_active_list[max_new_active_list_len];
    uint32_t next_active_list_index = 0;

    for (uint32_t i = 0; i < active_list_len; i++) {
        nodeptr active_node = active_list[i];
        if (Type(active_node) == CELL) {
            if (accept(active_node, current_node_size, current_node_midpoint)) {
                cell_ll_entry_t *new_entry = &free_enties[free_entry_index++];
                new_entry->priv = cell_list_tail;
                cell_list_tail = new_entry;
                new_entry->index = (cellptr) active_node - celltab;
            } else {
                for (nodeptr q = More(active_node); q != Next(active_node); q = Next(q)) {
                    assert(next_active_list_index < max_new_active_list_len);
                    next_active_list[next_active_list_index++] = q;
                }
            }
        } else {
            if (active_node != current_node) {
                cell_ll_entry_t *new_entry = &free_enties[free_entry_index++];
                new_entry->priv = body_list_tail;
                body_list_tail = new_entry;
                new_entry->index = (bodyptr) active_node - bodytab;
            }
        }
    }

    if (next_active_list_index > 0) {
        walksub(next_active_list, next_active_list_index,
                current_node, current_node_size, current_node_midpoint,
                cell_list_tail, body_list_tail, depth);
    } else {
        if (Type(current_node) != BODY) {
            error("walktree: recursion terminated with cell\n");
        }
        //gravsum((bodyptr) current_node, cell_list_tail, body_list_tail);
        cuda_gravsum((bodyptr) current_node, cell_list_tail, body_list_tail);
    }
}


#if defined(QUICKSCAN)

/*
 * ACCEPT: quick criterion accepts any cell not touching cell p.
 */

local bool accept(nodeptr c, real psize, vector pmid)
{
    real p15, dk;

    p15 = ((real) 1.5) * psize;                 /* premultiply cell size    */
    dk = Pos(c)[0] - pmid[0];                   /* find distance to midpnt  */
    if (ABS(dk) > p15)                          /* if c does not touch p    */
        return (TRUE);                          /* then accept interaction  */
    dk = Pos(c)[1] - pmid[1];                   /* find distance to midpnt  */
    if (ABS(dk) > p15)                          /* if c does not touch p    */
        return (TRUE);                          /* then accept interaction  */
    dk = Pos(c)[2] - pmid[2];                   /* find distance to midpnt  */
    if (ABS(dk) > p15)                          /* if c does not touch p    */
        return (TRUE);                          /* then accept interaction  */
    return (FALSE);                             /* else do not accept it    */
}

#else

/*
 * ACCEPT: standard criterion accepts cell if its critical radius
 * does not intersect cell p, and also imposes above condition.
 */

local bool accept(nodeptr c, real psize, vector pmid) {
    real dmax, dsq, dk;
    int k;

    dmax = psize; /* init maximum distance    */
    dsq = 0.0; /* and squared min distance */
    for (k = 0; k < NDIM; k++) {
        /* loop over space dims     */
        dk = Pos(c)[k] - pmid[k]; /* form distance to midpnt  */
        if (dk < 0) /* and get absolute value   */
            dk = -dk;
        if (dk > dmax) /* keep track of max value  */
            dmax = dk;
        dk -= ((real) 0.5) * psize; /* allow for size of cell   */
        if (dk > 0)
            dsq += dk * dk; /* sum min dist to cell ^2  */
    }
    return (dsq > Rcrit2(c) && /* test angular criterion   */
            dmax > ((real) 1.5) * psize); /* and adjacency criterion  */
}

#endif

/*
 * SUMNODE: add up body-node interactions.
 */

local void sumnode(cell_ll_entry_t *node_list_tail,
                   vector pos0, real *phi0, vector acc0, short type) {
    nodeptr p;
    real eps2, dr2, drab, phi_p, mr3i;
    vector dr;

    eps2 = eps * eps; /* avoid extra multiplys    */
    while (node_list_tail->priv != NULL) {
        /* loop over node list      */
        if (type == BODY) {
            p = (nodeptr)(bodytab + node_list_tail->index);
        } else {
            p = (nodeptr)(celltab + node_list_tail->index);
        }
        node_list_tail = node_list_tail->priv;
        DOTPSUBV(dr2, dr, Pos(p), pos0); /* compute separation       */
        /* and distance squared     */
        dr2 += eps2; /* add standard softening   */
        drab = sqrtf(dr2); /* form scalar "distance"   */
        phi_p = Mass(p) / drab; /* get partial potential    */
        *phi0 -= phi_p; /* decrement tot potential  */
        mr3i = phi_p / dr2; /* form scale factor for dr */
        ADDMULVS(acc0, dr, mr3i); /* sum partial acceleration */
    }
}

/*
 * SUMCELL: add up body-cell interactions.
 */

local void sumcell(cell_ll_entry_t *cell_list_tail,
                   vector pos0, real *phi0, vector acc0) {
    cellptr p;
    real eps2, dr2, drab, phi_p, mr3i, drqdr, dr5i, phi_q;
    vector dr, qdr;

    eps2 = eps * eps;
    while (cell_list_tail != NULL) {
        /* loop over node list      */
        p = celltab + cell_list_tail->index;
        cell_list_tail = cell_list_tail->priv;
        DOTPSUBV(dr2, dr, Pos(p), pos0); /* do mono part of force    */
        dr2 += eps2;
        drab = sqrtf(dr2);
        phi_p = Mass(p) / drab;
        mr3i = phi_p / dr2;
        DOTPMULMV(drqdr, qdr, Quad(p), dr); /* do quad part of force    */
        dr5i = ((real) 1.0) / (dr2 * dr2 * drab);
        phi_q = ((real) 0.5) * dr5i * drqdr;
        *phi0 -= phi_p + phi_q; /* add mono and quad pot    */
        mr3i += ((real) 5.0) * phi_q / dr2;
        ADDMULVS2(acc0, dr, mr3i, qdr, -dr5i); /* add mono and quad acc    */
    }
}

local void gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail) {
    vector pos0, acc0;
    real phi0;

    SETV(pos0, Pos(current_body)); /* copy position of body    */
    phi0 = 0.0; /* init total potential     */
    CLRV(acc0); /* and total acceleration   */
    if (usequad) /* if using quad moments    */
        sumcell(cell_list_tail, pos0, &phi0, acc0);
        /* sum cell forces w quads  */
    else /* not using quad moments   */
        sumnode(cell_list_tail, pos0, &phi0, acc0, CELL);
    /* sum cell forces wo quads */
    sumnode(body_list_tail, pos0, &phi0, acc0, BODY);
    /* sum forces from bodies   */
    Phi(current_body) = phi0; /* store total potential    */
    SETV(Acc(current_body), acc0); /* and total acceleration   */
    current_body->updated = TRUE; /* mark body as done        */
    //TODO: Do we want to keep track of this? Would need synchronization logic if so.
    //nbbcalc += interact + actlen - bptr;        /* count body-body forces   */
    //nbccalc += cptr - interact;                 /* count body-cell forces   */
}


