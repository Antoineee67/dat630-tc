#include "treedefs.h"
#include <cuda.h>
#include <mathfns.h>
#include <treecode.h>
#include <vector>
#include <vectdefs.h>
#include <vectmath.h>


#undef Update
#undef global

//#include "helper_math.h"
//#include <thrust/reduce.h>
//#include<thrust/execution_policy.h>
//#include <thrust/device_ptr.h>

typedef struct
{
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
    std::vector<size_t> body_cell_index_array_size; //byte size of above arrays
    uint32_t** device_body_cell_index_array_pointer_list; //Array to hold the device body cell index arrays on the device.
    std::vector<uint32_t> total_count; //length of above arrays.
    uint32_t* device_total_count; //pointer to device of above array;
    std::vector<bodyptr> current_body;

    vector* current_body_pos; //The vector position of the current body
    size_t current_body_pos_size; //Ugly solution, temporary.
    vector* device_current_body_pos; //Same on the device.

    real* device_phi_out_list; //Our out arrays.
    vector* device_acc_out_list;
};

static cudaStream_t localCudaStream;

static real* device_body_cell_mass_list;

static vector* device_body_cell_pos_list;

static cuda_grav_pack_soa cuda_grav_pack_list;

void new_cuda_grav_pack(cuda_grav_pack_soa* self, uint32_t* _body_cell_index_array, uint32_t* _device_body_cell_index_array, size_t _body_cell_index_array_size,
                        uint32_t _total_count, bodyptr _current_body)
{
    self->body_cell_index_array.push_back(_body_cell_index_array);
    self->device_body_cell_index_array.push_back(_device_body_cell_index_array);
    self->body_cell_index_array_size.push_back(_body_cell_index_array_size);
    self->total_count.push_back(_total_count);
    self->current_body.push_back(_current_body);
    SETV(self->current_body_pos[self->current_body_pos_size], Pos(_current_body));
    self->current_body_pos_size++;
}


__global__ void cuda_node_calc_kernel(real eps, uint32_t** body_cell_index_array_pointer, uint32_t* total_count_array,
                                      vector* body_cell_pos_list, real* body_cell_mass_list, real* phi_out_list, vector* acc_out_list, size_t maxSize, vector* currentBodypos)
{
    uint32_t workNumber = blockIdx.x * blockDim.x + threadIdx.x; //What body the gpu should work on corresponds to one element in the struct of arrays.

    if (workNumber > maxSize)
        return;
    uint32_t total_count = total_count_array[workNumber];
    uint32_t* body_cell_index_array = body_cell_index_array_pointer[workNumber];

    real dr2 = 0;
    vector dr;
    CLRV(dr);

    real local_phi0 = 0.0;
    vector local_acc0;
    CLRV(local_acc0);
    real eps2 = eps * eps;

    for (int i = 0; i < total_count; i++)
    {
        uint32_t loop_body = body_cell_index_array[i];
        DOTPSUBV(dr2, dr, body_cell_pos_list[loop_body], currentBodypos[workNumber]);

        dr2 += eps2;
        real drab = rsqrt(dr2);
        real phi_p = body_cell_mass_list[loop_body] / drab;
        local_phi0 -= phi_p;
        real mr3i = phi_p / dr2;
        ADDMULVS(local_acc0, dr, mr3i);
    }

    phi_out_list[workNumber] = local_phi0;
    SETV(acc_out_list[workNumber], local_acc0);
}

void cuda_gravsum(bodyptr current_body, cell_ll_entry_t* cell_list_tail, cell_ll_entry_t* body_list_tail)
{
    uint32_t cell_count = 0;
    uint32_t body_count = 0;
    cell_ll_entry_t* curr_list_entry;

    curr_list_entry = cell_list_tail;
    while (curr_list_entry->priv != NULL)
    {
        cell_count++;
        curr_list_entry = curr_list_entry->priv;
    }

    curr_list_entry = body_list_tail;

    while (curr_list_entry->priv != NULL)
    {
        body_count++;
        curr_list_entry = curr_list_entry->priv;
    }


    uint32_t* body_cell_index_array = (uint32_t*)malloc((body_count + cell_count) * sizeof(uint32_t));


    curr_list_entry = body_list_tail;
    for (int i = 0; i < body_count; i++)
    {
        body_cell_index_array[i] = curr_list_entry->index;
        curr_list_entry = curr_list_entry->priv;
    }



    curr_list_entry = cell_list_tail;
    for (int i = 0; i < cell_count; i++)
    {
        body_cell_index_array[i + body_count] = (curr_list_entry->index) + nbody;
        //+nbody to get correct offset for device_body_cell_pos_list? (Look in initalize and update_body_cell_data)
        curr_list_entry = curr_list_entry->priv;
    }




    uint32_t total_count = body_count + cell_count;
    uint32_t* device_body_cell_index_list;
    cudaMalloc(&device_body_cell_index_list, total_count * sizeof(uint32_t));


    new_cuda_grav_pack(&cuda_grav_pack_list, body_cell_index_array, device_body_cell_index_list, total_count * sizeof(uint32_t), total_count, current_body);
}

//Will send cuda_grav_pack_list to gpu and calculate.
void cuda_gravsum_dispatch()
{
    size_t bodiesToProcess = cuda_grav_pack_list.total_count.size();


    cudaMalloc(&cuda_grav_pack_list.device_total_count, sizeof(uint32_t) * bodiesToProcess);
    cudaMemcpy(cuda_grav_pack_list.device_total_count, cuda_grav_pack_list.total_count.data(),
               bodiesToProcess*sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_grav_pack_list.device_body_cell_index_array_pointer_list, sizeof(uint32_t*) * bodiesToProcess);
    cudaMemcpy(cuda_grav_pack_list.device_body_cell_index_array_pointer_list, cuda_grav_pack_list.device_body_cell_index_array.data(),
               bodiesToProcess*sizeof(uint32_t*) , cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_grav_pack_list.device_current_body_pos, sizeof(vector) * bodiesToProcess);
    cudaMemcpy(cuda_grav_pack_list.device_current_body_pos, cuda_grav_pack_list.current_body_pos,
                bodiesToProcess*sizeof(vector), cudaMemcpyHostToDevice);


    //cudaMemcpyAttributes attrList[] = {cudaMemcpyAttributes{cudaMemcpySrcAccessOrderAny, {}, {}, cudaMemcpyHostToDevice}};
    //size_t attrIndx[] = {0};

    //cudaMemcpyBatchAsync(cuda_grav_pack_list.device_body_cell_index_array.data(), cuda_grav_pack_list.body_cell_index_array.data(),
    //     cuda_grav_pack_list.body_cell_index_array_size.data(), bodiesToProcess, attrList,
    //     attrIndx, 1, &failIndx, localCudaStream
    //     );
    //printf("%lu", failIndx);

    for (int i = 0; i < bodiesToProcess; i++)
    {
        cudaMemcpy(cuda_grav_pack_list.device_body_cell_index_array[i], cuda_grav_pack_list.body_cell_index_array[i],
                   cuda_grav_pack_list.body_cell_index_array_size[i], cudaMemcpyDeviceToHost);
    }


    cudaMalloc(&cuda_grav_pack_list.device_phi_out_list, bodiesToProcess * sizeof(real));
    cudaMalloc(&cuda_grav_pack_list.device_acc_out_list, bodiesToProcess * sizeof(vector));

    //Start kernel
    int blocksize = 256;
    int nrBlocks = (bodiesToProcess + blocksize - 1) / blocksize;


    cuda_node_calc_kernel<<<nrBlocks, blocksize>>>(eps, cuda_grav_pack_list.device_body_cell_index_array_pointer_list,
                                                   cuda_grav_pack_list.device_total_count,
                                                   device_body_cell_pos_list,
                                                   device_body_cell_mass_list,
                                                   cuda_grav_pack_list.device_phi_out_list,
                                                   cuda_grav_pack_list.device_acc_out_list,
                                                   bodiesToProcess,
                                                   cuda_grav_pack_list.device_current_body_pos);

    real* phi_out_list = (real*)malloc(bodiesToProcess * sizeof(real));
    vector* acc_out_list = (vector*)malloc(bodiesToProcess * sizeof(vector));

    cudaMemcpy(phi_out_list, cuda_grav_pack_list.device_phi_out_list, bodiesToProcess * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(acc_out_list, cuda_grav_pack_list.device_acc_out_list, bodiesToProcess * sizeof(vector), cudaMemcpyDeviceToHost);

    //Apply phi0 and acc on bodies
    for (int i = 0; i < bodiesToProcess; i++)
    {
        Phi(cuda_grav_pack_list.current_body[i]) = phi_out_list[i];
        SETV(Acc(cuda_grav_pack_list.current_body[i]), acc_out_list[i]);
        cuda_grav_pack_list.current_body[i]->updated = TRUE;

        cudaFree(cuda_grav_pack_list.device_body_cell_index_array[i]);

        free(cuda_grav_pack_list.body_cell_index_array[i]);
    }
    //Free cuda memory
    cudaFree(cuda_grav_pack_list.device_phi_out_list);
    cudaFree(cuda_grav_pack_list.device_acc_out_list);
    cudaFree(cuda_grav_pack_list.device_body_cell_index_array_pointer_list);
    cudaFree(cuda_grav_pack_list.device_total_count);

    free(phi_out_list);
    free(acc_out_list);

    //cudaStreamSynchronize(localCudaStream);
    cuda_grav_pack_list.body_cell_index_array.clear();
    cuda_grav_pack_list.current_body.clear();
    cuda_grav_pack_list.device_body_cell_index_array.clear();
    cuda_grav_pack_list.body_cell_index_array_size.clear();
    cuda_grav_pack_list.total_count.clear();
    free(cuda_grav_pack_list.current_body_pos);
    cudaFree(cuda_grav_pack_list.device_current_body_pos);

    cuda_grav_pack_list.current_body_pos = (vector*) malloc(nbody*sizeof(vector)*2);
    cuda_grav_pack_list.current_body_pos_size = 0;
}


void cuda_gravsum_init()
{
    //cudaStreamCreate(&localCudaStream);
    printf("!!!!!!!!!! Only run on singlethread, struct of array is not thread safe");

    cuda_grav_pack_list.current_body_pos = (vector*) malloc(nbody*sizeof(vector)*2);
    cuda_grav_pack_list.current_body_pos_size = 0;

    cudaMalloc(&device_body_cell_mass_list, nbody * 2 * sizeof(real));
    cudaMalloc(&device_body_cell_pos_list, nbody * 2 * sizeof(vector));


    real body_mass_list[nbody];

    for (int i = 0; i < nbody; i++)
    {
        body_mass_list[i] = Mass(&bodytab[i]);
    }

    cudaMemcpy(device_body_cell_mass_list, body_mass_list, nbody * sizeof(real), cudaMemcpyHostToDevice);
}

void cuda_update_body_cell_data()
{



    vector body_cell_pos_list[nbody + ncell];

    for (int i = 0; i < nbody; i++)
    {
        SETV(body_cell_pos_list[i], Pos(&bodytab[i]));
    }

    for (int i = 0; i < ncell; i++)
    {
        SETV(body_cell_pos_list[i + nbody], Pos(&celltab[i]));
    }

    cudaMemcpy(device_body_cell_pos_list, body_cell_pos_list, (nbody + ncell) * sizeof(vector), cudaMemcpyHostToDevice);

    real cell_mass_list[ncell];

    for (int i = 0; i < ncell; i++)
    {
        cell_mass_list[i] = Mass(&celltab[i]);
    }
    cudaMemcpy(device_body_cell_mass_list + nbody, cell_mass_list, ncell * sizeof(real), cudaMemcpyHostToDevice);
}
