
#include "treedefs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <treecode.h>
#include <vectdefs.h>
#include <vectmath.h>

typedef struct {
    vector acc;
    real phi;
} body_result;

static real* device_body_mass_list;

static vector* device_body_pos_list;



__global__ void cuda_gravsum_kernel(uint32_t current_body, uint32_t* body_index_list, real* body_mass_list, vector* body_pos_list, body_result* result)
{

    //------------------------- Sumnode --------------------------//
    cellptr p;
    real eps2, dr2, drab, phi_p, mr3i, drqdr, dr5i, phi_q;
    vector dr, qdr;
    vector pos0, acc0;
    real phi0;

    eps2 = eps * eps; /* avoid extra multiplys    */

    //------------------------- Sumcell ---------------------------//

    //---------------------- Gravsum -----------------------//
    SETV(pos0, Pos(current_body)); /* copy position of body    */
    phi0 = 0.0;                    /* init total potential     */
    CLRV(acc0);                    /* and total acceleration   */
    while (body_list_tail->priv != NULL)
    {
        /* loop over node list      */
        p = &body_list_tail->cell;
        body_list_tail = body_list_tail->priv;
        DOTPSUBV(dr2, dr, Pos(p), pos0); /* compute separation       */
        /* and distance squared     */
        dr2 += eps2;              /* add standard softening   */
        drab = rsqrt(dr2);        /* form scalar "distance"   */
        phi_p = Mass(p) / drab;   /* get partial potential    */
        *phi0 -= phi_p;           /* decrement tot potential  */
        mr3i = phi_p / dr2;       /* form scale factor for dr */
        ADDMULVS(acc0, dr, mr3i); /* sum partial acceleration */
    }
    /* sum cell forces wo quads */
    while (body_list_tail->priv != NULL)
    {
        /* loop over node list      */
        p = &body_list_tail->cell;
        body_list_tail = body_list_tail->priv;
        DOTPSUBV(dr2, dr, Pos(p), pos0); /* compute separation       */
        /* and distance squared     */
        dr2 += eps2;              /* add standard softening   */
        drab = rsqrt(dr2);        /* form scalar "distance"   */
        phi_p = Mass(p) / drab;   /* get partial potential    */
        *phi0 -= phi_p;           /* decrement tot potential  */
        mr3i = phi_p / dr2;       /* form scale factor for dr */
        ADDMULVS(acc0, dr, mr3i); /* sum partial acceleration */
    }

    /* sum forces from bodies   */
    Phi(current_body) = phi0;      /* store total potential    */
    SETV(Acc(current_body), acc0); /* and total acceleration   */
    current_body->updated = TRUE;  /* mark body as done        */
    // TODO: Do we want to keep track of this? Would need synchronization logic if so.
    // nbbcalc += interact + actlen - bptr;        /* count body-body forces   */
    // nbccalc += cptr - interact;                 /* count body-cell forces   */
}


void cuda_gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail) {

    uint32_t current_body_index = current_body - bodytab;

    uint32_t cell_list_length = 0;
    cell_ll_entry_t *curr_list_entry = cell_list_tail;
    while (curr_list_entry->priv != NULL)
    {
        cell_list_length++;
        curr_list_entry = curr_list_entry->priv;
    }

    uint32_t cell_index_array[cell_list_length];
    curr_list_entry = cell_list_tail;

    for (int i = 0; i < cell_list_length; i++)
    {
        cell_index_array[i] = curr_list_entry->index;
        curr_list_entry = curr_list_entry->priv;
    }


    uint32_t body_list_length = 0;
    curr_list_entry = body_list_tail;
    while (curr_list_entry->priv != NULL)
    {
        body_list_length++;
        curr_list_entry = curr_list_entry->priv;
    }

    uint32_t body_index_array[body_list_length];
    curr_list_entry = body_list_tail;
    for (int i = 0; i < body_list_length; i++)
    {
        body_index_array[i] = curr_list_entry->index;
        curr_list_entry = curr_list_entry->priv;
    }

    uint32_t* device_cell_index_list;
    uint32_t* device_body_index_list;

    cudaMalloc(&device_cell_index_list, cell_list_length * sizeof(uint32_t));
    cudaMalloc(&device_body_index_list, body_list_length * sizeof(uint32_t));

    cudaMemcpy(device_cell_index_list, cell_index_array, cell_list_length * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_body_index_list, body_index_array, body_list_length * sizeof(uint32_t), cudaMemcpyHostToDevice);

    body_result* result;
    cudaMalloc(&result,  sizeof(body_result));

    //TODO
    cuda_gravsum_kernel<<<1, 1>>>(current_body_index, device_body_index_list, device_body_mass_list, device_body_pos_list, result);

    current_body->phi = result->phi;
    SETV(current_body->acc, result->acc);
    current_body->updated = TRUE;

    cudaFree(device_cell_index_list);
    cudaFree(device_body_index_list);
    cudaFree(result);

}

void cuda_gravsum_init()
{
    cudaMalloc(&device_body_mass_list, nbody * sizeof(real));
    cudaMalloc(&device_body_pos_list, nbody * sizeof(vector));

    real body_mass_list[nbody];

    for (int i = 0; i < nbody; i++)
    {
        body_mass_list[i] = Mass(&bodytab[i]);
    }

    cudaMemcpy(device_body_mass_list, body_mass_list, nbody * sizeof(real), cudaMemcpyHostToDevice);
}

void cuda_update_body_pos_list()
{
    vector body_pos_list[nbody];

    for (int i = 0; i < nbody; i++)
    {
        SETV(body_pos_list[i], Pos(&bodytab[i]));
    }

    cudaMemcpy(device_body_pos_list, body_pos_list, nbody * sizeof(vector), cudaMemcpyHostToDevice);
}

