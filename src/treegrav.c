/****************************************************************************/
/* TREEGRAV.C: routines to compute gravity. Public routines: gravcalc().    */
/* Copyright (c) 2001 by Joshua E. Barnes, Honolulu, Hawai`i.               */
/****************************************************************************/

#include "stdinc.h"
#include "mathfns.h"
#include "vectmath.h"
#include "treedefs.h"
#include "treecode.h"
#include "x86_64-linux-gnu/mpich/mpi.h"
//#include "mpi.h"


/* Local routines to perform force calculations. */

local void walktree(nodeptr *, nodeptr *, cellptr, cellptr,
                    nodeptr, real, vector);
local bool accept(nodeptr, real, vector);
local void walksub(nodeptr *, nodeptr *, cellptr, cellptr,
                   nodeptr, real, vector);
local void gravsum(bodyptr, cellptr, cellptr);
local void sumnode(cellptr, cellptr, vector, real *, vector);
local void sumcell(cellptr, cellptr, vector, real *, vector);

local void init_mpi();
// local void walktree_order(nodeptr *, nodeptr *, cellptr, cellptr,
//                     nodeptr, real, vector, bodyptr *, 
//                     size_t *, size_t *, size_t *, size_t *);
// local void walksub_order(nodeptr *, nodeptr *, cellptr, cellptr,
//                    nodeptr, real, vector, bodyptr *, 
//                    size_t *, size_t *, size_t *, size_t *);
local void walktree_order(nodeptr *, nodeptr *, cellptr, cellptr,
    nodeptr, real, vector);
local void walksub_order(nodeptr *, nodeptr *, cellptr, cellptr,
   nodeptr, real, vector);
local void work_partition(size_t *, size_t *, size_t *);

/* Lists of active nodes and interactions. */

#if !defined(FACTIVE)
#  define FACTIVE  0.75                         /* active list fudge factor */
#endif

local int actlen;                               /* length as allocated      */

local nodeptr *active;                          /* list of nodes tested     */

local cellptr interact;                         /* list of interactions     */

local bodyptr *ordered_btab;

local size_t *work_sum_bb;

local size_t *work_sum_bc;

local size_t *partition;

local size_t *body_count;

local real *mpi_acc = NULL;

local real *mpi_phi = NULL;

local real *mpi_acc_local;

local real *mpi_phi_local;


/*
 * GRAVCALC: perform force calculation on all particles.
 */

void gravcalc(void)
{
    double cpustart;
    vector rmid;

    actlen = FACTIVE * 216 * tdepth;            /* estimate list length     */
#if !defined(QUICKSCAN)
    actlen = actlen * rpow(theta, -2.5);        /* allow for opening angle  */
#endif


    active = (nodeptr *) allocate(actlen * sizeof(nodeptr));
    interact = (cellptr) allocate(actlen * sizeof(cell));
    ordered_btab = (bodyptr *) allocate(nbody * sizeof(nodeptr));
    work_sum_bb = (size_t*) allocate(nbody * sizeof(size_t));
    work_sum_bc = (size_t*) allocate(nbody * sizeof(size_t));
    partition = (size_t*) allocate(mpi_numproc * sizeof(size_t));
    body_count = (size_t*) allocate(sizeof(size_t));

    cpustart = cputime();                       /* record time, less alloc  */
    actmax = nbbcalc = nbccalc = 0;             /* zero cumulative counters */
    active[0] = (nodeptr) root;                 /* initialize active list   */
    CLRV(rmid);                                 /* set center of root cell  */
    *body_count = 0;

    walktree_order(active, active + 1, interact, interact + actlen,
            (nodeptr) root, rsize, rmid);
    if ((*body_count) != nbody){
        error("Wrong body count!\n");
    }

    nbbcalc = work_sum_bb[nbody-1];
    nbccalc = work_sum_bc[nbody-1];

    work_partition(partition, work_sum_bb, work_sum_bc);
    //printf("At rank = %d, partition = %d, %d, %d, %d\n", mpi_rank, partition[0],partition[1],partition[2],partition[3]);

    *body_count = 0;


    walktree(active, active + 1, interact, interact + actlen,
                (nodeptr) root, rsize, rmid);      /* scan tree, update forces */
        
    
    size_t start = partition[mpi_rank];
    size_t end = (mpi_rank == mpi_numproc - 1)? nbody : partition[mpi_rank+1];
    mpi_acc_local = (real *) allocate((end-start) * NDIM * sizeof(real));
    mpi_phi_local = (real *) allocate((end-start) * sizeof(real));
    size_t i;
    for (i = start; i < end; i++){
        mpi_phi_local[i-start] = Phi(ordered_btab[i]);
        if (NDIM == 3){
            mpi_acc_local[(i-start)*NDIM] = Acc(ordered_btab[i])[0];
            mpi_acc_local[(i-start)*NDIM+1] = Acc(ordered_btab[i])[1];
            mpi_acc_local[(i-start)*NDIM+2] = Acc(ordered_btab[i])[2];
        }
        else if (NDIM == 2){
            mpi_acc_local[(i-start)*NDIM] = Acc(ordered_btab[i])[0];
            mpi_acc_local[(i-start)*NDIM+1] = Acc(ordered_btab[i])[1];
        }
    }


    mpi_acc = (real *) allocate(nbody * NDIM * sizeof(real));
    mpi_phi = (real *) allocate(nbody * sizeof(real));

    int recvcounts_acc[mpi_numproc], recvcounts_phi[mpi_numproc];
    int displs_acc[mpi_numproc], displs_phi[mpi_numproc];

    for (i = 0; i < mpi_numproc; i++){
        size_t s = partition[i];
        size_t e = (i == mpi_numproc - 1)? nbody : partition[i+1];
        recvcounts_phi[i] = e - s;
        displs_phi[i] = partition[i];
        recvcounts_acc[i] = (e - s) * NDIM;
        displs_acc[i] = partition[i] * NDIM;
    }

    if (sizeof(real) == sizeof(float)){
        MPI_Allgatherv(mpi_acc_local, (end-start)*NDIM, MPI_FLOAT, mpi_acc, recvcounts_acc, displs_acc, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(mpi_phi_local, (end-start), MPI_FLOAT, mpi_phi, recvcounts_phi, displs_phi, MPI_FLOAT, MPI_COMM_WORLD);
    }
    else if (sizeof(real) == sizeof(double)){
        MPI_Allgatherv(mpi_acc_local, (end-start)*NDIM, MPI_DOUBLE, mpi_acc, recvcounts_acc, displs_acc, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(mpi_phi_local, (end-start), MPI_DOUBLE, mpi_phi, recvcounts_phi, displs_phi, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    
    
    for (i = 0; i < nbody; i++){
        Phi(ordered_btab[i]) = mpi_phi[i];
        if (NDIM == 3){
            Acc(ordered_btab[i])[0] = mpi_acc[i*NDIM]; 
            Acc(ordered_btab[i])[1] = mpi_acc[i*NDIM+1]; 
            Acc(ordered_btab[i])[2] = mpi_acc[i*NDIM+2]; 
        }
        else if (NDIM == 2){
            Acc(ordered_btab[i])[0] = mpi_acc[i*NDIM]; 
            Acc(ordered_btab[i])[1] = mpi_acc[i*NDIM+1]; 
        }
    }
    


    cpuforce = cputime() - cpustart;            /* store CPU time w/o alloc */

    free(active);
    free(interact);
    free(ordered_btab);
    free(work_sum_bb);
    free(work_sum_bc);
    free(partition);
    free(body_count);

    free(mpi_acc_local);
    free(mpi_phi_local);
    free(mpi_acc);
    free(mpi_phi);


    
}

local void init_mpi(){
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
}

local void work_partition(size_t * partition, size_t * work_sum_bb, size_t * work_sum_bc){

    size_t total_work = work_sum_bb[nbody-1] + work_sum_bc[nbody-1];
    size_t work_for_each = total_work / mpi_numproc;

    size_t i;
    size_t r = 1;
    partition[0] = 0;

    for (i = 0; i < nbody; i++){
        if (work_sum_bb[i] + work_sum_bc[i] >= work_for_each*r){
            partition[r] = i;
            if (r == mpi_numproc - 1){
                break;
            }
            r++;
        }
    }
}


local void walktree_order(nodeptr *aptr, nodeptr *nptr, cellptr cptr, cellptr bptr,
                    nodeptr p, real psize, vector pmid)
{

    nodeptr *np, *ap, q;
    int actsafe;

    if (Update(p)) {                            /* are new forces needed?   */
        np = nptr;                              /* start new active list    */
        actsafe = actlen - NSUB;                /* leave room for NSUB more */
        for (ap = aptr; ap < nptr; ap++)        /* loop over active nodes   */
            if (Type(*ap) == CELL) {            /* is this node a cell?     */
                if (accept(*ap, psize, pmid)) { /* does it pass the test?   */
                    Mass(cptr) = Mass(*ap);     /* copy to interaction list */
                    SETV(Pos(cptr), Pos(*ap));
                    SETM(Quad(cptr), Quad(*ap));
                    cptr++;                     /* and bump cell array ptr  */
                } else {                        /* else it fails the test   */
                    if (np - active >= actsafe) /* check list has room      */
                        error("walktree: active list overflow\n");
                    for (q = More(*ap); q != Next(*ap); q = Next(q))
                                                /* loop over all subcells   */
                        *np++= q;               /* put on new active list   */
                }
            } else                              /* else this node is a body */
                if (*ap != p) {                 /* if not self-interaction  */
                    --bptr;                     /* bump body array ptr      */
                    Mass(bptr) = Mass(*ap);     /* and copy data to array   */
                    SETV(Pos(bptr), Pos(*ap));
                }
        actmax = MAX(actmax, np - active);      /* keep track of max active */
        if (np != nptr)                         /* if new actives listed    */
            walksub_order(nptr, np, cptr, bptr, p, psize, pmid);
                                                /* then visit next level    */
        else {                                  /* else no actives left, so */
            if (Type(p) != BODY)                /* must have found a body   */
                error("walktree: recursion terminated with cell\n");
            ordered_btab[(*body_count)] = (bodyptr) p;
            if ((*body_count) == 0){
                work_sum_bb[(*body_count)] = interact + actlen - bptr;
                work_sum_bc[(*body_count)] = cptr - interact;
            }
            else{
                work_sum_bb[(*body_count)] = work_sum_bb[(*body_count)-1] + interact + actlen - bptr;
                work_sum_bc[(*body_count)] = work_sum_bc[(*body_count)-1] + cptr - interact;
            }
            (*body_count)++;
            // gravsum((bodyptr) p, cptr, bptr);   /* sum force on the body    */
        }
    }

}



local void walksub_order(nodeptr *nptr, nodeptr *np, cellptr cptr, cellptr bptr,
                   nodeptr p, real psize, vector pmid)
{
    real poff;
    nodeptr q;
    int k;
    vector nmid;

    poff = psize / 4;                           /* precompute mid. offset   */
    if (Type(p) == CELL) {                      /* fanout over descendents  */
        for (q = More(p); q != Next(p); q = Next(q)) {
                                                /* loop over all subcells   */
            for (k = 0; k < NDIM; k++)          /* locate each's midpoint   */
                nmid[k] = pmid[k] + (Pos(q)[k] < pmid[k] ? - poff : poff);
            walktree_order(nptr, np, cptr, bptr, q, psize / 2, nmid);
                                                /* recurse on subcell       */
        }
    } else {                                    /* extend virtual tree      */
        for (k = 0; k < NDIM; k++)              /* locate next midpoint     */
            nmid[k] = pmid[k] + (Pos(p)[k] < pmid[k] ? - poff : poff);
        walktree_order(nptr, np, cptr, bptr, p, psize / 2, nmid);
                                                /* and search next level    */
    }
}



/*
 * WALKTREE: do a complete walk of the tree, building the interaction
 * list level-by-level and computing the resulting force on each body.
 */

local void walktree(nodeptr *aptr, nodeptr *nptr, cellptr cptr, cellptr bptr,
                    nodeptr p, real psize, vector pmid)
{

    nodeptr *np, *ap, q;
    int actsafe;



    if (Update(p)) {                            /* are new forces needed?   */
        np = nptr;                              /* start new active list    */
        actsafe = actlen - NSUB;                /* leave room for NSUB more */
        for (ap = aptr; ap < nptr; ap++)        /* loop over active nodes   */
            if (Type(*ap) == CELL) {            /* is this node a cell?     */
                if (accept(*ap, psize, pmid)) { /* does it pass the test?   */
                    Mass(cptr) = Mass(*ap);     /* copy to interaction list */
                    SETV(Pos(cptr), Pos(*ap));
                    SETM(Quad(cptr), Quad(*ap));
                    cptr++;                     /* and bump cell array ptr  */
                } else {                        /* else it fails the test   */
                    if (np - active >= actsafe) /* check list has room      */
                        error("walktree: active list overflow\n");
                    for (q = More(*ap); q != Next(*ap); q = Next(q))
                                                /* loop over all subcells   */
                        *np++= q;               /* put on new active list   */
                }
            } else                              /* else this node is a body */
                if (*ap != p) {                 /* if not self-interaction  */
                    --bptr;                     /* bump body array ptr      */
                    Mass(bptr) = Mass(*ap);     /* and copy data to array   */
                    SETV(Pos(bptr), Pos(*ap));
                }
        actmax = MAX(actmax, np - active);      /* keep track of max active */
        if (np != nptr)                         /* if new actives listed    */
            walksub(nptr, np, cptr, bptr, p, psize, pmid);
                                                /* then visit next level    */
        else {                                  /* else no actives left, so */
            if (Type(p) != BODY)                /* must have found a body   */
                error("walktree: recursion terminated with cell\n");
            size_t start = partition[mpi_rank];
            size_t end = (mpi_rank == mpi_numproc - 1)? nbody : partition[mpi_rank+1];
            if ((*body_count) >= start && (*body_count) < end){
                gravsum((bodyptr) p, cptr, bptr);   /* sum force on the body    */
            }
            (*body_count)++;
            //gravsum((bodyptr) p, cptr, bptr);   /* sum force on the body    */
        }
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

local bool accept(nodeptr c, real psize, vector pmid)
{
    real dmax, dsq, dk;
    int k;

    dmax = psize;                               /* init maximum distance    */
    dsq = 0.0;                                  /* and squared min distance */
    for (k = 0; k < NDIM; k++) {                /* loop over space dims     */
        dk = Pos(c)[k] - pmid[k];               /* form distance to midpnt  */
        if (dk < 0)                             /* and get absolute value   */
            dk = - dk;
        if (dk > dmax)                          /* keep track of max value  */
            dmax = dk;
        dk -= ((real) 0.5) * psize;             /* allow for size of cell   */
        if (dk > 0)
            dsq += dk * dk;                     /* sum min dist to cell ^2  */
    }
    return (dsq > Rcrit2(c) &&                  /* test angular criterion   */
              dmax > ((real) 1.5) * psize);     /* and adjacency criterion  */
}

#endif

/*
 * WALKSUB: test next level's active list against subnodes of p.
 */

local void walksub(nodeptr *nptr, nodeptr *np, cellptr cptr, cellptr bptr,
                   nodeptr p, real psize, vector pmid)
{
    real poff;
    nodeptr q;
    int k;
    vector nmid;

    poff = psize / 4;                           /* precompute mid. offset   */
    if (Type(p) == CELL) {                      /* fanout over descendents  */
        for (q = More(p); q != Next(p); q = Next(q)) {
                                                /* loop over all subcells   */
            for (k = 0; k < NDIM; k++)          /* locate each's midpoint   */
                nmid[k] = pmid[k] + (Pos(q)[k] < pmid[k] ? - poff : poff);
            walktree(nptr, np, cptr, bptr, q, psize / 2, nmid);
                                                /* recurse on subcell       */
        }
    } else {                                    /* extend virtual tree      */
        for (k = 0; k < NDIM; k++)              /* locate next midpoint     */
            nmid[k] = pmid[k] + (Pos(p)[k] < pmid[k] ? - poff : poff);
        walktree(nptr, np, cptr, bptr, p, psize / 2, nmid);
                                                /* and search next level    */
    }
}

/*
 * GRAVSUM: compute gravitational field at body p0.
 */

local void gravsum(bodyptr p0, cellptr cptr, cellptr bptr)
{
    vector pos0, acc0;
    real phi0;

    SETV(pos0, Pos(p0));                        /* copy position of body    */
    phi0 = 0.0;                                 /* init total potential     */
    CLRV(acc0);                                 /* and total acceleration   */
    if (usequad)                                /* if using quad moments    */
        sumcell(interact, cptr, pos0, &phi0, acc0);
                                                /* sum cell forces w quads  */
    else                                        /* not using quad moments   */
        sumnode(interact, cptr, pos0, &phi0, acc0);
                                                /* sum cell forces wo quads */
    sumnode(bptr, interact + actlen, pos0, &phi0, acc0);
                                                /* sum forces from bodies   */
    Phi(p0) = phi0;                             /* store total potential    */
    SETV(Acc(p0), acc0);                        /* and total acceleration   */
    // nbbcalc += interact + actlen - bptr;        /* count body-body forces   */
    // nbccalc += cptr - interact;                 /* count body-cell forces   */
}

/*
 * SUMNODE: add up body-node interactions.
 */

local void sumnode(cellptr start, cellptr finish,
                   vector pos0, real *phi0, vector acc0)
{
    cellptr p;
    real eps2, dr2, drab, phi_p, mr3i;
    vector dr;

    eps2 = eps * eps;                           /* avoid extra multiplys    */
    for (p = start; p < finish; p++) {          /* loop over node list      */
        DOTPSUBV(dr2, dr, Pos(p), pos0);        /* compute separation       */
                                                /* and distance squared     */
        dr2 += eps2;                            /* add standard softening   */
        drab = rsqrt(dr2);                      /* form scalar "distance"   */
        phi_p = Mass(p) / drab;                 /* get partial potential    */
        *phi0 -= phi_p;                         /* decrement tot potential  */
        mr3i = phi_p / dr2;                     /* form scale factor for dr */
        ADDMULVS(acc0, dr, mr3i);               /* sum partial acceleration */
    }
}

/*
 * SUMCELL: add up body-cell interactions.
 */

local void sumcell(cellptr start, cellptr finish,
                   vector pos0, real *phi0, vector acc0)
{
    cellptr p;
    real eps2, dr2, drab, phi_p, mr3i, drqdr, dr5i, phi_q;
    vector dr, qdr;

    eps2 = eps * eps;
    for (p = start; p < finish; p++) {          /* loop over node list      */
        DOTPSUBV(dr2, dr, Pos(p), pos0);        /* do mono part of force    */
        dr2 += eps2;
        drab = rsqrt(dr2);
        phi_p = Mass(p) / drab;
        mr3i = phi_p / dr2;
        DOTPMULMV(drqdr, qdr, Quad(p), dr);     /* do quad part of force    */
        dr5i = ((real) 1.0) / (dr2 * dr2 * drab);
        phi_q = ((real) 0.5) * dr5i * drqdr;
        *phi0 -= phi_p + phi_q;                 /* add mono and quad pot    */
        mr3i += ((real) 5.0) * phi_q / dr2;
        ADDMULVS2(acc0, dr, mr3i, qdr, -dr5i);  /* add mono and quad acc    */
    }
}
