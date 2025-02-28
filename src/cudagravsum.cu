#include "treedefs.h"
#include <cuda.h>
#include <treecode.h>
#include <vector>
#include <vectdefs.h>
#include <vectmath.h>


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

/*struct cuda_grav_pack
{
    uint32_t *body_cell_index_array; //ptr to body & cell array with indices that refer to correct body/cell.
    uint32_t *device_body_cell_index_array; //equivelent pointer but for the cuda device.
    uint32_t total_count; //length of above arrays.
    bodyptr current_body;

};*/

struct cuda_grav_pack_soa
{
    std::vector<uint32_t*> body_cell_index_array; //ptr to body & cell array with indices that refer to correct body/cell.
    std::vector<uint32_t*> device_body_cell_index_array; //equivelent pointer but for the cuda device.
    std::vector<size_t> device_body_cell_index_array_size; //byte size of above arrays
    uint32_t** device_body_cell_index_array_pointer_list; //Array to hold the device body cell index arrays.
    std::vector<uint32_t> total_count; //length of above arrays.
    uint32_t* device_total_count; //pointer to device of above array;
    std::vector<bodyptr> current_body;
    real *device_phi_out_list; //Our out arrays.
    vector *device_acc_out_list;
};

static cudaStream_t localCudaStream;

static real *device_body_cell_mass_list;

static cuda_vector *device_body_cell_pos_list;

// static  cuda_grav_pack_soa cuda_grav_pack_list;


static thrust::host_vector<uint32_t> h_interact_vecs;

static thrust::host_vector<uint32_t> h_offset;

static thrust::host_vector<size_t> h_bodies_to_process;


// void new_cuda_grav_pack(cuda_grav_pack_soa *self, uint32_t *_body_cell_index_array, uint32_t *_device_body_cell_index_array, size_t _device_body_cell_index_array_size, uint32_t _total_count, bodyptr _current_body)
// {
//     self->body_cell_index_array.push_back(_body_cell_index_array);
//     self->device_body_cell_index_array.push_back(_device_body_cell_index_array);
//     self->device_body_cell_index_array_size.push_back(_device_body_cell_index_array_size);
//     self->total_count.push_back(_total_count);
//     self->current_body.push_back(_current_body);
// }





__global__ void cuda_node_calc_kernel(real eps2, uint32_t* interact_lists, uint32_t* offset, size_t* bodies_to_process,
    cuda_vector* pos_list, real* mass_list, real *out_phi_list, cuda_vector* out_acc_list, size_t n_bodies){

    uint32_t list_index = blockIdx.x*blockDim.x + threadIdx.x; //What body the gpu should work on corresponds to one element in the struct of arrays.

    if (list_index >= n_bodies)
        return;
    uint32_t end = offset[list_index];
    uint32_t start;
    if (list_index == 0){
        start = 0;
    }
    else{
        start = offset[list_index-1]+1;
    }

    real dr2, drab, phi_p, mr3i;
    //vector dr;
    cuda_vector dr;
    real local_phi0 = 0;
    cuda_vector local_acc0;
    CLR_CUDA_VECTOR(local_acc0);
    //CLRV(local_acc0);

    size_t current_body_index = bodies_to_process[list_index];

    for (uint32_t i = start; i < end; i++)
    {   
        size_t loop_body_index = interact_lists[i];
        // DOTPSUBV(dr2, dr, pos_list[loop_body_index], pos_list[current_body_index]);
        dr = pos_list[loop_body_index] - pos_list[current_body_index];
        dr2 = dot(dr, dr);
        dr2 += eps2;
        drab = rsqrt(dr2);
        phi_p = mass_list[loop_body_index]/drab;
        local_phi0 -= phi_p;
        mr3i = phi_p/dr2;
        //ADDMULVS(local_acc0, dr, mr3i);
        local_acc0 += dr * mr3i;
    }

    out_phi_list[current_body_index] = local_phi0;
    out_acc_list[current_body_index] = local_acc0;
}

void cuda_gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail) {

    uint32_t cell_count = 0;
    cell_ll_entry_t *curr_list_entry = cell_list_tail;
    while (curr_list_entry->priv != nullptr) {
        h_interact_vecs.push_back(curr_list_entry->index + nbody);
        cell_count++;
        curr_list_entry = curr_list_entry->priv;
    }

    uint32_t body_count = 0;
    curr_list_entry = body_list_tail;
    while (curr_list_entry->priv != nullptr) {
        h_interact_vecs.push_back(curr_list_entry->index);
        body_count++;
        curr_list_entry = curr_list_entry->priv;
    }

    h_offset.push_back(h_interact_vecs.size()-1); // = end
    h_bodies_to_process.push_back((size_t)(current_body-bodytab));

}

//Will send cuda_grav_pack_list to gpu and calculate.
void cuda_gravsum_dispatch()
{
    size_t nBodiesToProcess = h_bodies_to_process.size();

    thrust::device_vector<uint32_t> d_interact_vecs;
    thrust::copy(thrust::cuda::par.on(localCudaStream), h_interact_vecs.begin(), h_interact_vecs.end(), d_interact_vecs.begin());
    thrust::device_vector<uint32_t> d_offset; 
    thrust::copy(thrust::cuda::par.on(localCudaStream), h_offset.begin(), h_offset.end(), d_offset.begin());
    thrust::device_vector<size_t> d_bodies_to_process; 
    thrust::copy(thrust::cuda::par.on(localCudaStream), h_bodies_to_process.begin(), h_bodies_to_process.end(), d_bodies_to_process.begin());
    thrust::device_vector<real> d_out_phi(nBodiesToProcess, 0);
    thrust::device_vector<cuda_vector> d_out_acc(nBodiesToProcess);

    uint32_t* d_interact_vecs_raw = thrust::raw_pointer_cast(d_interact_vecs.data());
    uint32_t* d_offset_raw = thrust::raw_pointer_cast(d_offset.data());
    size_t* d_bodies_to_process_raw = thrust::raw_pointer_cast(d_bodies_to_process.data());
    real* d_out_phi_raw = thrust::raw_pointer_cast(d_out_phi.data());
    cuda_vector* d_out_acc_raw = thrust::raw_pointer_cast(d_out_acc.data());


    //Start kernel
    int blocksize = 256;
    int nrGrids = (nBodiesToProcess + blocksize - 1)/blocksize;


    cuda_node_calc_kernel<<<nrGrids, blocksize, 0, localCudaStream>>>(eps2, 
        d_interact_vecs_raw, // 1D vector which links all interact lists
        d_offset_raw, // offset
        d_bodies_to_process_raw, 
        device_body_cell_pos_list, // pos
        device_body_cell_mass_list, // mass
        d_out_phi_raw, // phi
        d_out_acc_raw, // acc
        nBodiesToProcess);

    thrust::host_vector<real> h_out_phi;
    thrust::copy(thrust::cuda::par.on(localCudaStream), d_out_phi.begin(), d_out_phi.end(), h_out_phi.begin());
    thrust::host_vector<cuda_vector> h_out_acc;
    thrust::copy(thrust::cuda::par.on(localCudaStream), d_out_acc.begin(), d_out_acc.end(), h_out_acc.begin());

    

    //Apply phi0 and acc on bodies
    for (int i = 0; i < nBodiesToProcess; i++)
    {
        bodyptr current_bptr = bodytab+h_bodies_to_process[i];
        Phi(current_bptr) = h_out_phi[i];
        SETV(Acc(current_bptr), CUDA_VECTOR_TO_VECTOR(h_out_acc[i]));
        current_bptr->updated = TRUE;
    }
    //Free cuda memory
    // cudaFreeAsync(cuda_grav_pack_list.device_phi_out_list, localCudaStream);
    // cudaFreeAsync(cuda_grav_pack_list.device_acc_out_list, localCudaStream);
    // cudaFreeAsync(cuda_grav_pack_list.device_body_cell_index_array_pointer_list, localCudaStream);
    // cudaFreeAsync(cuda_grav_pack_list.device_total_count, localCudaStream);

    cudaStreamSynchronize(localCudaStream);


}



void cuda_gravsum_init() {
    //TODO: Guard against/allow multiple calls?
    cudaStreamCreate(&localCudaStream);
    printf("!!!!!!!!!! Only run on singlethread, struct of array is not thread safe\n");


    cudaMalloc(&device_body_cell_mass_list, nbody * 2 * sizeof(real));
    cudaMalloc(&device_body_cell_pos_list, nbody * 2 * sizeof(cuda_vector));



    real body_mass_list[nbody];

    for (int i = 0; i < nbody; i++) {
        body_mass_list[i] = Mass(&bodytab[i]);
    }

    cudaMemcpy(device_body_cell_mass_list, body_mass_list, nbody * sizeof(real), cudaMemcpyHostToDevice);
}

void cuda_update_body_cell_data() {
    vector body_cell_pos_list[nbody + ncell];

    for (int i = 0; i < nbody; i++) {
        SETV(body_cell_pos_list[i], Pos(&bodytab[i]));
    }

    for (int i = 0; i < ncell; i++) {
         SETV(body_cell_pos_list[i + nbody], Pos(&celltab[i]));
    }

    cudaMemcpy(device_body_cell_pos_list, body_cell_pos_list, nbody * sizeof(cuda_vector), cudaMemcpyHostToDevice);

    real cell_mass_list[ncell];

    for (int i = 0; i < ncell; i++) {
        cell_mass_list[i] = Mass(&celltab[i]);
    }
    cudaMemcpy(device_body_cell_mass_list + nbody, cell_mass_list, ncell * sizeof(real), cudaMemcpyHostToDevice);

    h_interact_vecs.clear();
    h_offset.clear();
    h_bodies_to_process.clear();
}
