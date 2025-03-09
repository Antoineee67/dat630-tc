#include "treedefs.h"
#include <cuda.h>
#include <treecode.h>
#include <vector>
#include <vectdefs.h>
#include <vectmath.h>
#include <cmath>
#include "helper_math.h"
#include "cudagravsum.cuh"

#undef Update
#undef global



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

#define N_CUDA_STREAMS 4
#define BLOCK_SIZE 256

static cudaStream_t localCudaStreams[N_CUDA_STREAMS];

static real *device_body_cell_mass_list;

static cuda_vector *device_body_cell_pos_list;


static std::vector<uint32_t> h_interact_vecs;

static std::vector<uint32_t> h_offset;

static std::vector<size_t> h_bodies_to_process;

real* d_out_phi_raw;

cuda_vector* d_out_acc_raw;

int idle_stream = 0;



__global__ void cuda_node_calc_kernel(real eps2, uint32_t* interact_lists, uint32_t* offset, size_t* bodies_to_process,
    cuda_vector* pos_list, real* mass_list, real *out_phi_list, cuda_vector* out_acc_list, size_t n_bodies){

    uint32_t list_index = blockIdx.x*blockDim.x + threadIdx.x; 
    int tx = threadIdx.x;

    if (list_index >= n_bodies)
        return;
    uint32_t end = offset[list_index];
    uint32_t start;
    if (list_index == 0){
        start = 0;
    }
    else{
        start = offset[list_index-1];
    }

    real dr2, drab, phi_p, mr3i;
    //vector dr;
    cuda_vector dr;
    real local_phi0 = 0;
    cuda_vector local_acc0;
    //CLRV(local_acc0);
    CLR_CUDA_VECTOR(local_acc0);
    
    size_t current_body_index = bodies_to_process[list_index];

    for (uint32_t i = start; i < end; i++)
    {   
        size_t loop_body_index = interact_lists[i];
        // DOTPSUBV(dr2, dr, pos_list[loop_body_index], pos_list[current_body_index]);
        dr = pos_list[loop_body_index] - pos_list[current_body_index];
        dr2 = dot(dr, dr);
        dr2 += eps2;
        drab = sqrt(dr2);
        phi_p = mass_list[loop_body_index]/drab;
        local_phi0 -= phi_p;
        mr3i = phi_p/dr2;
        //ADDMULVS(local_acc0, dr, mr3i);
        local_acc0 += dr * mr3i;
    }

    out_phi_list[current_body_index] = local_phi0;
    out_acc_list[current_body_index] = local_acc0;
}

__global__ void cuda_node_reduction_kernel(real eps2, uint32_t* interact_lists, uint32_t* offset, size_t* bodies_to_process,
    cuda_vector* pos_list, real* mass_list, real *out_phi_list, cuda_vector* out_acc_list, size_t n_bodies){
    
    int tx = threadIdx.x;
    size_t list_index = blockIdx.x;
    size_t current_body_index = bodies_to_process[list_index];

    uint32_t end = offset[list_index];
    uint32_t start;
    if (list_index == 0){
        start = 0;
    }
    else{
        start = offset[list_index-1];
    }

    __shared__ real local_phi_p[BLOCK_SIZE];
    __shared__ cuda_vector local_acc[BLOCK_SIZE];
    real dr2, drab, phi_p, mr3i;
    cuda_vector dr;
    real block_sum_phi = 0;
    cuda_vector block_sum_acc;
    CLR_CUDA_VECTOR(block_sum_acc);
    
    uint32_t size = ceil((float)(end-start)/blockDim.x);
    for (uint32_t i = 0; i < size; i++){
        uint32_t block_start = i * blockDim.x + start;

        if (tx + block_start < end){
            size_t loop_body_index = interact_lists[tx+block_start];
            dr = pos_list[loop_body_index] - pos_list[current_body_index];
            dr2 = dot(dr, dr);
            dr2 += eps2;
            drab = sqrt(dr2);
            phi_p = mass_list[loop_body_index]/drab;
            local_phi_p[tx] = -phi_p;
            mr3i = phi_p/dr2;
            //ADDMULVS(local_acc0, dr, mr3i);
            local_acc[tx] = dr * mr3i;       
        }
        __syncthreads();

        for (int s = blockDim.x/2; s > 0; s >>= 1){
            if (tx < s && tx + block_start + s < end){
                local_phi_p[tx] += local_phi_p[tx+s];
                local_acc[tx] += local_acc[tx+s];
            }
            __syncthreads();
        }

        if (tx == 0){
            block_sum_phi += local_phi_p[0];
            block_sum_acc += local_acc[0];
        }
    }

    if (tx == 0){
        out_phi_list[current_body_index] = block_sum_phi;
        out_acc_list[current_body_index] = block_sum_acc;
    }
    
}



void cuda_gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail) {

    cell_ll_entry_t *curr_list_entry = cell_list_tail;
    while (curr_list_entry->priv != nullptr) {       
        h_interact_vecs.push_back(curr_list_entry->index + nbody);;
        curr_list_entry = curr_list_entry->priv;
    }

    curr_list_entry = body_list_tail;
    while (curr_list_entry->priv != nullptr) {     
        h_interact_vecs.push_back(curr_list_entry->index);
        curr_list_entry = curr_list_entry->priv;
    } 

    h_offset.push_back(h_interact_vecs.size()); // = end
    h_bodies_to_process.push_back((size_t)(current_body-bodytab));

    if (h_bodies_to_process.size()==BLOCK_SIZE*256){
        cuda_gravsum_dispatch();
        h_interact_vecs.clear();
        h_bodies_to_process.clear();
        h_offset.clear();
    }

    

}

//Will send cuda_grav_pack_list to gpu and calculate.
void cuda_gravsum_dispatch()
{

    size_t nBodiesToProcess = h_bodies_to_process.size();

    uint32_t* h_interact_vecs_raw = h_interact_vecs.data();
    uint32_t* h_offset_raw = h_offset.data();
    size_t* h_bodies_to_process_raw = h_bodies_to_process.data();
    // real h_out_phi_raw[nBodiesToProcess];
    // cuda_vector h_out_acc_raw[nBodiesToProcess]; 


    uint32_t* d_interact_vecs_raw;
    cudaMalloc(&d_interact_vecs_raw, sizeof(uint32_t)*h_interact_vecs.size());
    uint32_t* d_offset_raw;
    cudaMalloc(&d_offset_raw, sizeof(uint32_t)*nBodiesToProcess);
    size_t* d_bodies_to_process_raw;
    cudaMalloc(&d_bodies_to_process_raw, sizeof(size_t)*nBodiesToProcess);
    // real* d_out_phi_raw;
    // cudaMalloc(&d_out_phi_raw, sizeof(real)*nBodiesToProcess);
    // cuda_vector* d_out_acc_raw;
    // cudaMalloc(&d_out_acc_raw, sizeof(cuda_vector)*nBodiesToProcess);

    // real* d_mass_raw;
    // cudaMalloc(&d_mass_raw, sizeof(real)*nBodiesToProcess);
    // cuda_vector* d_pos_raw;
    // cudaMalloc(&d_pos_raw, sizeof(cuda_vector)*nBodiesToProcess);
    
    // real h_mass_raw[nBodiesToProcess];
    // cuda_vector h_pos_raw[nBodiesToProcess];

    // for (size_t i = 0; i < nBodiesToProcess; i++){
    //     bodyptr current_bptr = bodytab + h_bodies_to_process[i];
    //     h_mass_raw[]
    // }



    cudaMemcpyAsync(d_offset_raw, h_offset_raw, sizeof(uint32_t)*nBodiesToProcess, 
                    cudaMemcpyHostToDevice, localCudaStreams[idle_stream]);
    cudaMemcpyAsync(d_bodies_to_process_raw, h_bodies_to_process_raw, sizeof(size_t)*nBodiesToProcess, 
                    cudaMemcpyHostToDevice, localCudaStreams[idle_stream]);
    cudaMemcpyAsync(d_interact_vecs_raw, h_interact_vecs_raw, sizeof(uint32_t)*h_interact_vecs.size(), 
                    cudaMemcpyHostToDevice, localCudaStreams[idle_stream]);
                
    cudaStreamSynchronize(localCudaStreams[idle_stream]);
    
    int blocksize = BLOCK_SIZE;
    int nrGrids = (nBodiesToProcess + blocksize - 1)/blocksize;

    // cuda_node_calc_kernel<<<nrGrids, blocksize, 0, localCudaStreams[idle_stream]>>>(eps2, 
    //     d_interact_vecs_raw, // 1D vector which links all interact lists
    //     d_offset_raw, // offset for each body in 1D vector
    //     d_bodies_to_process_raw, 
    //     device_body_cell_pos_list, // pos
    //     device_body_cell_mass_list, // mass
    //     d_out_phi_raw, // phi
    //     d_out_acc_raw, // acc
    //     nBodiesToProcess); // offset for each stream
    cuda_node_reduction_kernel<<<nBodiesToProcess, blocksize, 0, localCudaStreams[idle_stream]>>>(eps2, 
        d_interact_vecs_raw, // 1D vector which links all interact lists
        d_offset_raw, // offset for each body in 1D vector
        d_bodies_to_process_raw, 
        device_body_cell_pos_list, // pos
        device_body_cell_mass_list, // mass
        d_out_phi_raw, // phi
        d_out_acc_raw, // acc
        nBodiesToProcess); // offset for each stream

    cudaFreeAsync(d_interact_vecs_raw, localCudaStreams[idle_stream]);
    cudaFreeAsync(d_offset_raw, localCudaStreams[idle_stream]);
    cudaFreeAsync(d_bodies_to_process_raw, localCudaStreams[idle_stream]);

    idle_stream = (idle_stream+1) % N_CUDA_STREAMS;


    // cudaMemcpyAsync(h_out_phi_raw+lower, d_out_phi_raw+lower, sizeof(real)*width, 
    //                 cudaMemcpyDeviceToHost, localCudaStreams[i]);
    // cudaMemcpyAsync(h_out_acc_raw+lower, d_out_acc_raw+lower, sizeof(cuda_vector)*width, 
    //                 cudaMemcpyDeviceToHost, localCudaStreams[i]);

    

     


    // for (size_t i = 0; i < nBodiesToProcess; i++)
    // {
    //     bodyptr current_bptr = bodytab + h_bodies_to_process[i];
    //     size_t current_body_index = h_bodies_to_process[i];
    //     Phi(current_bptr) = h_out_phi_raw[i];
    //     SETV(Acc(current_bptr), CUDA_VECTOR_TO_VECTOR(h_out_acc_raw[i]));
    //     current_bptr->updated = TRUE;
    // }

    
    

}

void cuda_dispatch_last_batch(){

    if (h_bodies_to_process.size()>0){
        cuda_gravsum_dispatch();
    }
}


void cuda_collect_result(){

    for (int i = 0; i < N_CUDA_STREAMS; i++){
        cudaStreamSynchronize(localCudaStreams[i]);
    }

    //real h_out_phi_raw[nbody];
    std::vector<real> h_out_phi_raw(nbody);
    //cuda_vector h_out_acc_raw[nbody];
    std::vector<cuda_vector> h_out_acc_raw(nbody);

    cudaMemcpy(h_out_phi_raw.data(), d_out_phi_raw, sizeof(real)*nbody, 
                    cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_acc_raw.data(), d_out_acc_raw, sizeof(cuda_vector)*nbody, 
                    cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nbody; i++)
    {
        bodyptr current_bptr = bodytab + i;
        Phi(current_bptr) = h_out_phi_raw[i];
        SETV(Acc(current_bptr), CUDA_VECTOR_TO_VECTOR(h_out_acc_raw[i]));
        current_bptr->updated = TRUE;
    }
}



void cuda_gravsum_init() {
    //TODO: Guard against/allow multiple calls?
    //cudaStreamCreate(&localCudaStream);
    for (int i = 0; i < N_CUDA_STREAMS; i++){
        cudaStreamCreate(&localCudaStreams[i]);
    }
    // printf("!!!!!!!!!! Only run on singlethread, struct of array is not thread safe\n");


    cudaMalloc(&device_body_cell_mass_list, nbody * 2 * sizeof(real));
    cudaMalloc(&device_body_cell_pos_list, nbody * 2 * sizeof(cuda_vector));



    real body_mass_list[nbody];

    for (int i = 0; i < nbody; i++) {
        body_mass_list[i] = Mass(&bodytab[i]);
    }
    
    cudaMemcpy(device_body_cell_mass_list, body_mass_list, nbody * sizeof(real), cudaMemcpyHostToDevice);


    cudaMalloc(&d_out_phi_raw, sizeof(real)*nbody);
    cudaMalloc(&d_out_acc_raw, sizeof(cuda_vector)*nbody);

}

void cuda_update_body_cell_data() {

    // vector body_cell_pos_list[nbody + ncell];
    // cuda_vector body_cell_pos_list[nbody + ncell];
    std::vector<cuda_vector> body_cell_pos_list(nbody+ncell);

    for (int i = 0; i < nbody; i++) {
        //SETV(body_cell_pos_list[i], Pos(&bodytab[i]));
        body_cell_pos_list[i] = VECTOR_TO_CUDA_VECTOR(Pos(&bodytab[i]));
    }

    for (int i = 0; i < ncell; i++) {
        //SETV(body_cell_pos_list[i + nbody], Pos(&celltab[i]));
        body_cell_pos_list[i+nbody] = VECTOR_TO_CUDA_VECTOR(Pos(&celltab[i]));
    }


    cudaMemcpy(device_body_cell_pos_list, body_cell_pos_list.data(), (nbody+ncell) * sizeof(cuda_vector), cudaMemcpyHostToDevice);

    //real cell_mass_list[ncell];
    std::vector<real> cell_mass_list(ncell);

    for (int i = 0; i < ncell; i++) {
        cell_mass_list[i] = Mass(&celltab[i]);
    }
    cudaMemcpy(device_body_cell_mass_list + nbody, cell_mass_list.data(), ncell * sizeof(real), cudaMemcpyHostToDevice);


    h_interact_vecs.clear();
    h_offset.clear();
    h_bodies_to_process.clear();
}


void cuda_free_all(){
    cudaFree(device_body_cell_mass_list);
    cudaFree(device_body_cell_pos_list);
    cudaFree(d_out_acc_raw);
    cudaFree(d_out_phi_raw);
    for (int i = 0; i < N_CUDA_STREAMS; i++){
        cudaStreamDestroy(localCudaStreams[i]);
    }
}
