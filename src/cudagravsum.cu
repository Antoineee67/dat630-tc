#include "treedefs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <treecode.h>
#include <vectdefs.h>
#include <vectmath.h>

#undef Update
#undef global

#include "helper_math.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

typedef struct {
    vector acc;
    real phi;
} body_result;



#define EXPAND(x) x
#define CONCAT(a, b) a##b
#define MAKE_TYPE(base, dim) CONCAT(base, dim)

#ifndef DOUBLEPREC
#define cuda_vector MAKE_TYPE(float, NDIM)
#else
#define cuda_vector MAKE_TYPE(double, NDIM)
#endif


#if NDIM == 2
#define CUDA_VECTOR_TO_VECTOR(cv) ((vector){(cv).x, (cv).y})
#define VECTOR_TO_CUDA_VECTOR(v) ((cuda_vector){(v)[0], (v)[1]})
#elif NDIM == 3
#define CUDA_VECTOR_TO_VECTOR(cv) ((vector){(cv).x, (cv).y, (cv).z})
#define VECTOR_TO_CUDA_VECTOR(v) ((cuda_vector){(v)[0], (v)[1], (v)[2]})
#elif NDIM == 4
#define CUDA_VECTOR_TO_VECTOR(cv) ((vector){(cv).x, (cv).y, (cv).z, (cv).w})
#define VECTOR_TO_CUDA_VECTOR(v) ((cuda_vector){(v)[0], (v)[1], (v)[2], (v)[3]})
#else
#error "NDIM must be 2, 3, or 4"
#endif

static real *device_body_cell_mass_list;

static cuda_vector *device_body_cell_pos_list;



__global__ void cuda_node_calc_kernel(uint32_t current_body, real eps2, uint32_t *body_cell_index_list,
                                      real *body_cell_mass_list, cuda_vector *body_cell_pos_list, real *phi_out_list,
                                      cuda_vector *acc_out_list) {
    uint32_t tid = threadIdx.x;

    //DOTPSUBV(dr2, dr, body_cell_pos_list[body_cell_index_list[tid]], body_cell_pos_list[current_body]); /* compute separation       */
    cuda_vector dr = body_cell_pos_list[body_cell_index_list[tid]] - body_cell_pos_list[current_body];
    real dr2 = dot(dr, dr);

    /* and distance squared     */
    dr2 += eps2; /* add standard softening   */
    real drab = rsqrt(dr2); /* form scalar "distance"   */
    real phi_p = body_cell_mass_list[body_cell_index_list[tid]] / drab; /* get partial potential    */
    phi_out_list[tid] = -phi_p; /* store partial potential  */
    real mr3i = phi_p / dr2; /* form scale factor for dr */
    acc_out_list[tid] = dr * mr3i; // Calculate acceleration
}

void cuda_gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail) {
    uint32_t current_body_index = current_body - bodytab;

    uint32_t cell_count = 0;
    cell_ll_entry_t *curr_list_entry = cell_list_tail;
    while (curr_list_entry->priv != nullptr) {
        cell_count++;
        curr_list_entry = curr_list_entry->priv;
    }

    uint32_t body_count = 0;
    curr_list_entry = body_list_tail;
    while (curr_list_entry->priv != nullptr) {
        body_count++;
        curr_list_entry = curr_list_entry->priv;
    }

    uint32_t body_cell_index_array[body_count + cell_count];
    curr_list_entry = body_list_tail;
    for (int i = 0; i < body_count; i++) {
        body_cell_index_array[i] = curr_list_entry->index;
        curr_list_entry = curr_list_entry->priv;
    }
    curr_list_entry = cell_list_tail;
    for (int i = 0; i < cell_count; i++) {
        body_cell_index_array[i + body_count] = curr_list_entry->index;
        curr_list_entry = curr_list_entry->priv;
    }

    const uint32_t total_count = body_count + cell_count;

    uint32_t *device_body_cell_index_list;

    cudaMalloc(&device_body_cell_index_list,  total_count * sizeof(uint32_t));
    cudaMemcpy(device_body_cell_index_list, body_cell_index_array, total_count * sizeof(uint32_t), cudaMemcpyHostToDevice);

    real *device_phi_result_list;
    cuda_vector *device_acc_result_list;

    cudaMalloc(&device_phi_result_list, total_count * sizeof(real));
    cudaMalloc(&device_acc_result_list, total_count * sizeof(cuda_vector));


    //TODO: Look over the block and grid size
    cuda_node_calc_kernel<<<1, total_count>>>(current_body_index, eps2, device_body_cell_index_list, device_body_cell_mass_list, device_body_cell_pos_list, device_phi_result_list, device_acc_result_list);

    thrust::device_ptr<real> thrust_device_phi_result_list(device_phi_result_list);
    thrust::device_ptr<cuda_vector> thrust_device_acc_result_list(device_acc_result_list);

    real phi = thrust::reduce(thrust_device_phi_result_list, thrust_device_phi_result_list + total_count);
    cuda_vector acc = thrust::reduce(thrust_device_acc_result_list, thrust_device_acc_result_list + total_count);

    current_body->phi = phi;
    SETV(current_body->acc, CUDA_VECTOR_TO_VECTOR(acc));
    current_body->updated = TRUE;

    cudaFree(device_body_cell_index_list);
    cudaFree(device_phi_result_list);
    cudaFree(device_acc_result_list);
}

void cuda_gravsum_init() {
    //TODO: Guard against/allow multiple calls?
    cudaMalloc(&device_body_cell_mass_list, nbody * 2 * sizeof(real));
    cudaMalloc(&device_body_cell_pos_list, nbody * 2 * sizeof(cuda_vector));

    real body_mass_list[nbody];

    for (int i = 0; i < nbody; i++) {
        body_mass_list[i] = Mass(&bodytab[i]);
    }

    cudaMemcpy(device_body_cell_mass_list, body_mass_list, nbody * sizeof(real), cudaMemcpyHostToDevice);
}

void cuda_update_body_cell_data() {
    cuda_vector body_cell_pos_list[nbody + ncell];

    for (int i = 0; i < nbody; i++) {
        body_cell_pos_list[i] = VECTOR_TO_CUDA_VECTOR(Pos(&bodytab[i]));
    }

    for (int i = 0; i < ncell; i++) {
        body_cell_pos_list[i + nbody] = VECTOR_TO_CUDA_VECTOR(Pos(&celltab[i]));
    }

    cudaMemcpy(device_body_cell_pos_list, body_cell_pos_list, nbody * sizeof(cuda_vector), cudaMemcpyHostToDevice);

    real cell_mass_list[ncell];

    for (int i = 0; i < ncell; i++) {
        cell_mass_list[i] = Mass(&celltab[i]);
    }
    cudaMemcpy(device_body_cell_mass_list + nbody, cell_mass_list, ncell * sizeof(real), cudaMemcpyHostToDevice);
}
