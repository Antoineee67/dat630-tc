#include "treedefs.h"
#include <cuda.h>
#include <treecode.h>
#include <vector>
#include <vectdefs.h>
#include <vectmath.h>
#include <cmath>


#undef Update
#undef global

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "helper_math.h"

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

static cudaStream_t localCudaStreams[N_CUDA_STREAMS];

static real *device_body_cell_mass_list;

static cuda_vector *device_body_cell_pos_list;


static thrust::host_vector<uint32_t> h_interact_vecs;

static thrust::host_vector<uint32_t> h_offset;

static thrust::host_vector<size_t> h_bodies_to_process;



__global__ void cuda_node_calc_kernel(real eps2, uint32_t* interact_lists, uint32_t* offset, size_t* bodies_to_process,
    cuda_vector* pos_list, real* mass_list, real *out_phi_list, cuda_vector* out_acc_list, size_t width, uint32_t interact_start){

    uint32_t list_index = blockIdx.x*blockDim.x + threadIdx.x; //What body the gpu should work on corresponds to one element in the struct of arrays.


    size_t current_body_index = bodies_to_process[list_index];

    // if (current_body_index == 0){
    //     printf("list_index = %d\n", list_index);
    //     printf("width = %d\n", width);
    //     for (int i = 0; i< width;i++){
    //         printf("hello from %d\n", current_body_index);
    //         printf("i = %d\n", i);
    //         printf("bi = %d\n", bodies_to_process[i]);
    //     }
        
    // }

    if (list_index >= width)
        return;
    uint32_t end = offset[list_index] - interact_start;

    // if (current_body_index == 0){
    //     printf("hello2\n");
    // }

    uint32_t start;
    if (list_index == 0){
        start = 0;
    }
    else{
        start = offset[list_index-1] - interact_start;
    }

    // if (current_body_index == 0){
    //     printf("hello3\n");
    // }

    real dr2, drab, phi_p, mr3i;
    //vector dr;
    cuda_vector dr;
    real local_phi0 = 0;
    cuda_vector local_acc0;
    //CLRV(local_acc0);
    CLR_CUDA_VECTOR(local_acc0);

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

    // if (current_body_index == 0){
    //     printf("hello4\n");
    // }

    out_phi_list[list_index] = local_phi0;
    out_acc_list[list_index] = local_acc0;

    // if (current_body_index == 0){
    //     printf("hello5\n");
    // }
}

bool first_call = true;

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
}

//Will send cuda_grav_pack_list to gpu and calculate.
void cuda_gravsum_dispatch()
{

    size_t nBodiesToProcess = h_bodies_to_process.size();

    thrust::host_vector<real> h_out_phi(nBodiesToProcess);
    thrust::host_vector<cuda_vector> h_out_acc(nBodiesToProcess);
    thrust::device_vector<uint32_t> d_interact_vecs(h_interact_vecs.size());
    thrust::device_vector<uint32_t> d_offset(nBodiesToProcess); 
    thrust::device_vector<size_t> d_bodies_to_process(nBodiesToProcess); 
    thrust::device_vector<real> d_out_phi(nBodiesToProcess, 0);
    thrust::device_vector<cuda_vector> d_out_acc(nBodiesToProcess);

    uint32_t* h_interact_vecs_raw = thrust::raw_pointer_cast(h_interact_vecs.data());
    uint32_t* h_offset_raw = thrust::raw_pointer_cast(h_offset.data());
    size_t* h_bodies_to_process_raw = thrust::raw_pointer_cast(h_bodies_to_process.data());
    real* h_out_phi_raw = thrust::raw_pointer_cast(h_out_phi.data());
    cuda_vector* h_out_acc_raw = thrust::raw_pointer_cast(h_out_acc.data()); 

    uint32_t* d_interact_vecs_raw = thrust::raw_pointer_cast(d_interact_vecs.data());
    uint32_t* d_offset_raw = thrust::raw_pointer_cast(d_offset.data());
    size_t* d_bodies_to_process_raw = thrust::raw_pointer_cast(d_bodies_to_process.data());
    real* d_out_phi_raw = thrust::raw_pointer_cast(d_out_phi.data());
    cuda_vector* d_out_acc_raw = thrust::raw_pointer_cast(d_out_acc.data()); 


    size_t chunk_size = nBodiesToProcess / N_CUDA_STREAMS;

    for (int i = 0; i < N_CUDA_STREAMS; i++){

        size_t lower = chunk_size * i;
        size_t upper = min(lower+chunk_size, nBodiesToProcess);
        size_t width = upper - lower;

        cudaMemcpyAsync(d_offset_raw+lower, h_offset_raw+lower, sizeof(uint32_t)*width, 
                        cudaMemcpyHostToDevice, localCudaStreams[i]);
        cudaMemcpyAsync(d_bodies_to_process_raw+lower, h_bodies_to_process_raw+lower, sizeof(size_t)*width, 
                        cudaMemcpyHostToDevice, localCudaStreams[i]);

        uint32_t interact_lower = (lower>0)? h_offset[lower-1] : 0;
        uint32_t interact_width = h_offset[upper-1] - interact_lower;

        cudaMemcpyAsync(d_interact_vecs_raw+interact_lower, h_interact_vecs_raw+interact_lower, sizeof(uint32_t)*interact_width, 
                        cudaMemcpyHostToDevice, localCudaStreams[i]);
        
        int blocksize = 256;
        int nrGrids = (width + blocksize - 1)/blocksize;
        
        // printf("copy success for stream %d\n", i);

        cuda_node_calc_kernel<<<nrGrids, blocksize, 0, localCudaStreams[i]>>>(eps2, 
            d_interact_vecs_raw+interact_lower, // 1D vector which links all interact lists
            d_offset_raw+lower, // offset
            d_bodies_to_process_raw+lower, 
            device_body_cell_pos_list, // pos
            device_body_cell_mass_list, // mass
            d_out_phi_raw+lower, // phi
            d_out_acc_raw+lower, // acc
            width,
            interact_lower); 
        
        // printf("kernel success for stream %d\n", i);

        cudaMemcpyAsync(h_out_phi_raw+lower, d_out_phi_raw+lower, sizeof(real)*width, 
                        cudaMemcpyDeviceToHost, localCudaStreams[i]);
        cudaMemcpyAsync(h_out_acc_raw+lower, d_out_acc_raw+lower, sizeof(cuda_vector)*width, 
                        cudaMemcpyDeviceToHost, localCudaStreams[i]);
        // printf("copy back success for stream %d\n", i);
    }

    for (int i = 0; i < N_CUDA_STREAMS; i++){
        cudaStreamSynchronize(localCudaStreams[i]);
    }
    

    // Apply phi0 and acc on bodies
    for (size_t i = 0; i < nBodiesToProcess; i++)
    {
        bodyptr current_bptr = bodytab + h_bodies_to_process[i];
        size_t current_body_index = h_bodies_to_process[i];
        Phi(current_bptr) = h_out_phi[i];
        SETV(Acc(current_bptr), CUDA_VECTOR_TO_VECTOR(h_out_acc[i]));
        current_bptr->updated = TRUE;
    }

    //printf("load success for stream\n");

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

}

void cuda_update_body_cell_data() {

    // vector body_cell_pos_list[nbody + ncell];
    cuda_vector body_cell_pos_list[nbody + ncell];

    for (int i = 0; i < nbody; i++) {
        //SETV(body_cell_pos_list[i], Pos(&bodytab[i]));
        body_cell_pos_list[i] = VECTOR_TO_CUDA_VECTOR(Pos(&bodytab[i]));
    }

    for (int i = 0; i < ncell; i++) {
        //SETV(body_cell_pos_list[i + nbody], Pos(&celltab[i]));
        body_cell_pos_list[i+nbody] = VECTOR_TO_CUDA_VECTOR(Pos(&celltab[i]));
    }

    cudaMemcpy(device_body_cell_pos_list, body_cell_pos_list, (nbody+ncell) * sizeof(cuda_vector), cudaMemcpyHostToDevice);

    real cell_mass_list[ncell];

    for (int i = 0; i < ncell; i++) {
        cell_mass_list[i] = Mass(&celltab[i]);
    }
    cudaMemcpy(device_body_cell_mass_list + nbody, cell_mass_list, ncell * sizeof(real), cudaMemcpyHostToDevice);


    h_interact_vecs.clear();
    h_offset.clear();
    h_bodies_to_process.clear();
}


void cuda_free_all(){
    cudaFree(device_body_cell_mass_list);
    cudaFree(device_body_cell_pos_list);
    for (int i = 0; i < N_CUDA_STREAMS; i++){
        cudaStreamDestroy(localCudaStreams[i]);
    }
}
