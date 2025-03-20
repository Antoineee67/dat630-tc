#ifndef CUDATREE_H
#define CUDATREE_H

#include "treedefs.h"


#undef global

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
#define CLR_CUDA_VECTOR(cv) ((cv) = {0, 0})
#elif NDIM == 3
#define CUDA_VECTOR_TO_VECTOR(cv) ((vector){(cv).x, (cv).y, (cv).z})
#define VECTOR_TO_CUDA_VECTOR(v) ((cuda_vector){(v)[0], (v)[1], (v)[2]})
#define CLR_CUDA_VECTOR(cv) ((cv) = {0, 0, 0})
#elif NDIM == 4
#define CUDA_VECTOR_TO_VECTOR(cv) ((vector){(cv).x, (cv).y, (cv).z, (cv).w})
#define VECTOR_TO_CUDA_VECTOR(v) ((cuda_vector){(v)[0], (v)[1], (v)[2], (v)[3]})
#define CLR_CUDA_VECTOR(cv) ((cv) = {0, 0, 0, 0})
#else
#error "NDIM must be 2, 3, or 4"
#endif

//#define BLOCK_SIZE 256

void cuda_copy_tree();

void cuda_tree_init();

void cuda_tree_free();

void cuda_tree_compute();

void cuda_tree_collect_result();

__global__ void cuda_tree_compute_kernel(real* mass_list, cuda_vector* pos_list, int32_t* child_list, 
    real* rcrit_list, real* phi_list, cuda_vector* acc_list, int nbody, real eps2, real rsize);

__device__ void cuda_tree_traverse(int32_t body_cell_index, int32_t curr_body_index, real* mass_list, 
    cuda_vector* pos_list, int32_t* child_list, real* rcrit_list, real* phi_list, cuda_vector* acc_list, 
    int nbody, real eps2);

__device__ bool cuda_accept(real rcrti, cuda_vector cell_pos, cuda_vector curr_body_pos);

__device__ bool cuda_accept2(real rcrti, cuda_vector cell_pos, real psize, cuda_vector pmid);
      
__device__ void cuda_tree_sumnode(int32_t body_cell_index, int32_t curr_body_index, real* mass_list, 
    cuda_vector* pos_list, real* phi_list, cuda_vector* acc_list, real eps2);

__device__ bool cuda_locate_within(cuda_vector cell_pos, cuda_vector body_pos, cuda_vector pmid);

#endif