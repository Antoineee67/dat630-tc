#include "treedefs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <vectdefs.h>
#include <vectmath.h>
#include <cmath>
#include "treecode.h"
#include "helper_math.h"
#include "cudatree.cuh"



real* d_mass_list;

cuda_vector* d_pos_list;

int32_t* d_child_list; 

real* d_rcrit_list;

real* d_phi_list;

cuda_vector* d_acc_list;

//int32_t* d_nbody;

void cuda_tree_init(){
    cudaMalloc(&d_mass_list, sizeof(real)*(nbody+ncell));
    cudaMalloc(&d_pos_list, sizeof(cuda_vector)*(nbody+ncell));
    cudaMalloc(&d_child_list, sizeof(int32_t)*(ncell*NSUB));
    cudaMalloc(&d_rcrit_list, sizeof(real)*ncell);
    cudaMalloc(&d_phi_list, sizeof(real)*nbody);
    cudaMalloc(&d_acc_list, sizeof(cuda_vector)*nbody);
}

void cuda_tree_free(){
    cudaFree(d_mass_list);
    cudaFree(d_pos_list);
    cudaFree(d_child_list);
    cudaFree(d_rcrit_list);
    cudaFree(d_phi_list);
    cudaFree(d_acc_list);
}

void cuda_copy_tree(){

    // To copy: mass, pos, child, rcrit
    
    std::vector<real> h_mass_list;

    std::vector<cuda_vector> h_pos_list;

    std::vector<int32_t> h_child_list; 

    std::vector<real> h_rcrit_list;

    for (int i = 0; i < nbody; i++) {
        h_mass_list.push_back(Mass(&bodytab[i]));
        h_pos_list.push_back(VECTOR_TO_CUDA_VECTOR(Pos(&bodytab[i])));
    }

    for (int i = 0; i < ncell; i++) {
        h_mass_list.push_back(Mass(&celltab[i]));
        h_pos_list.push_back(VECTOR_TO_CUDA_VECTOR(Pos(&celltab[i])));
        h_rcrit_list.push_back(Rcrit2(&celltab[i]));
        for (int j = 0; j < NSUB; j++){
            h_child_list.push_back(ChildIdx(&celltab[i])[j]);
        }
    }

    real* h_mass_list_raw = h_mass_list.data();
    cuda_vector* h_pos_list_raw = h_pos_list.data();
    int32_t* h_child_list_raw = h_child_list.data();
    real* h_rcrit_list_raw = h_rcrit_list.data();

    cudaMemcpy(d_mass_list, h_mass_list_raw, sizeof(real)*(nbody+ncell), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_list, h_pos_list_raw, sizeof(cuda_vector)*(nbody+ncell), cudaMemcpyHostToDevice);
    cudaMemcpy(d_child_list, h_child_list_raw, sizeof(int32_t)*(ncell*NSUB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rcrit_list, h_rcrit_list_raw, sizeof(real)*(ncell), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_nbody, &nbody, sizeof(int32_t), cudaMemcpyHostToDevice);

}

void cuda_tree_compute(){

    int block_size = cuda_blocksize;
    int grid_size = (nbody + block_size - 1)/block_size;
    
    cuda_tree_compute_kernel<<<grid_size, block_size>>>(d_mass_list, d_pos_list, d_child_list, d_rcrit_list, 
        d_phi_list, d_acc_list, nbody, eps2, rsize);

}

void cuda_tree_collect_result(){

    std::vector<real> h_out_phi_list(nbody);
    std::vector<cuda_vector> h_out_acc_list(nbody);

    cudaMemcpy(h_out_phi_list.data(), d_phi_list, sizeof(real)*nbody, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_acc_list.data(), d_acc_list, sizeof(cuda_vector)*nbody, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nbody; i++)
    {
        bodyptr current_bptr = bodytab + i;
        Phi(current_bptr) = h_out_phi_list[i];
        SETV(Acc(current_bptr), CUDA_VECTOR_TO_VECTOR(h_out_acc_list[i]));
        current_bptr->updated = TRUE;
    }
}



__global__ void cuda_tree_compute_kernel(real* mass_list, cuda_vector* pos_list, int32_t* child_list, 
    real* rcrit_list, real* phi_list, cuda_vector* acc_list, int nbody, real eps2, real rsize){
    
    uint32_t curr_body_index = blockIdx.x*blockDim.x + threadIdx.x; 

    if (curr_body_index >= nbody){
        return;
    }

    cuda_vector curr_body_pos = pos_list[curr_body_index];
    
    phi_list[curr_body_index] = 0;
    CLR_CUDA_VECTOR(acc_list[curr_body_index]);

    // cuda_tree_traverse(nbody, curr_body_index, mass_list, pos_list, child_list, rcrit_list, 
    //     phi_list, acc_list, nbody, eps2);

    // int32_t stack[128];
    // size_t top = 0;
    // stack[top++] = nbody;
        
    // while (top > 0){
    //     int32_t body_cell_index = stack[--top];
    //     if (body_cell_index < nbody){
    //         // calc on body
    //         cuda_tree_sumnode(body_cell_index, curr_body_index, mass_list, pos_list, phi_list, acc_list, eps2);
    //     }
    //     else{
    //         real rcrit = rcrit_list[body_cell_index-nbody];
    //         cuda_vector cell_pos = pos_list[body_cell_index];
    //         cuda_vector curr_body_pos = pos_list[curr_body_index];
    //         if (cuda_accept(rcrit, cell_pos, curr_body_pos)){
    //             // calc on cell
    //             cuda_tree_sumnode(body_cell_index, curr_body_index, mass_list, pos_list, phi_list, acc_list, eps2);
    //         }
    //         else{
    //             int32_t child_index;
    //             for (int i = 0; i < NSUB; i++){
    //                 child_index = child_list[(body_cell_index-nbody)*NSUB+i];
    //                 if (child_index >= 0){
    //                     stack[top++] = child_index;
    //                 }
    //             }
    //         }
    //     }
    // }

    int32_t max_active_len = 0.75*216*25;
    int32_t active_list[(int)(0.75*216*25)];
    active_list[0] = nbody;
    int32_t active_len = 1;
    int32_t active_start = 0;
    int32_t curr_node_index = nbody;
    real psize = rsize;
    cuda_vector pmid;
    CLR_CUDA_VECTOR(pmid);

    while(active_len > 0){
        int loop_start = active_start;
        int loop_end = active_start+active_len;
        active_start = active_start+active_len;
        active_len = 0;
        for (int i = loop_start; i < loop_end; i++){
            int32_t active_node_index = active_list[i];
            if (active_node_index < nbody){
                cuda_tree_sumnode(active_node_index, curr_body_index, mass_list, pos_list, phi_list, acc_list, eps2);
            }
            else{
                real rcrit = rcrit_list[active_node_index-nbody];
                cuda_vector cell_pos = pos_list[active_node_index];
                if (cuda_accept2(rcrit, cell_pos, psize, pmid)){
                    cuda_tree_sumnode(active_node_index, curr_body_index, mass_list, pos_list, phi_list, acc_list, eps2);
                }
                else{
                    int32_t child_index;
                    for (int j = 0; j < NSUB; j++){
                        child_index = child_list[(active_node_index-nbody)*NSUB+j];
                        if (child_index >= 0){
                            active_list[active_start+active_len] = child_index;
                            active_len++;
                        }
                    }
                }
            }
        }

        if (curr_node_index >= nbody){
            int32_t child_index;
            for (int i = 0; i < NSUB; i++){
                child_index = child_list[(curr_node_index-nbody)*NSUB+i];
                if (child_index >= 0){
                    cuda_vector child_pos = pos_list[child_index];
                    if (cuda_locate_within(child_pos, curr_body_pos, pmid)){
                        curr_node_index = child_index;
                        break;
                    }
                }
            }
        }

        cuda_vector curr_node_pos = pos_list[curr_node_index];
        real poff = psize/4;
        pmid.x += (curr_node_pos.x < pmid.x)? -poff : poff;
        pmid.y += (curr_node_pos.y < pmid.y)? -poff : poff;
        pmid.z += (curr_node_pos.z < pmid.z)? -poff : poff;
        psize /= 2;

    }
}

__device__ bool cuda_locate_within(cuda_vector cell_pos, cuda_vector body_pos, cuda_vector pmid){
    
    cuda_vector d_cell = cell_pos - pmid;
    cuda_vector d_body = body_pos - pmid;
    return ((d_cell.x*d_body.x >=0) && (d_cell.y*d_body.y >=0) && (d_cell.z*d_body.z >=0));
}



__device__ void cuda_tree_traverse(int32_t body_cell_index, int32_t curr_body_index, real* mass_list, 
    cuda_vector* pos_list, int32_t* child_list, real* rcrit_list, real* phi_list, cuda_vector* acc_list, 
    int nbody, real eps2){

    if (body_cell_index < nbody){
        // calc on body
        if (curr_body_index == 0){
            printf("calc on body\n");
        }
        cuda_tree_sumnode(body_cell_index, curr_body_index, mass_list, pos_list, phi_list, acc_list, eps2);
    }
    else{
        real rcrit = rcrit_list[body_cell_index-nbody];
        cuda_vector cell_pos = pos_list[body_cell_index];
        cuda_vector curr_body_pos = pos_list[curr_body_index];
        if (cuda_accept(rcrit, cell_pos, curr_body_pos)){
            // calc on cell
            if (curr_body_index == 0){
                printf("calc on cell\n");
            }
            cuda_tree_sumnode(body_cell_index, curr_body_index, mass_list, pos_list, phi_list, acc_list, eps2);
        }
        else{
            if (curr_body_index == 0){
                printf("not accept at %d\n", body_cell_index);
            }
            int32_t child_index;
            for (int i = 0; i < NSUB; i++){
                child_index = child_list[(body_cell_index-nbody)*NSUB+i];
                if (child_index >= 0){
                    cuda_tree_traverse(child_index, curr_body_index, mass_list, pos_list, 
                        child_list, rcrit_list, phi_list, acc_list, nbody, eps2);
                }
            }
        }
    }
}

__device__ bool cuda_accept(real rcrti, cuda_vector cell_pos, cuda_vector curr_body_pos){

    cuda_vector d = cell_pos - curr_body_pos;
    real dsq = d.x*d.x + d.y*d.y + d.z*d.z;
    return (dsq > rcrti);
}

__device__ bool cuda_accept2(real rcrti, cuda_vector cell_pos, real psize, cuda_vector pmid){

    real dmax, dsq;
    cuda_vector d = fabs(cell_pos - pmid);

    dmax = psize;
    if (d.x > dmax){
        dmax = d.x;
    }
    if (d.y > dmax){
        dmax = d.y;
    }
    if (d.z > dmax){
        dmax = d.z;
    }

    d -= ((real) 0.5) * psize;
    dsq = 0;
    if (d.x > 0){
        dsq += d.x*d.x;
    }
    if (d.y > 0){
        dsq += d.y*d.y;
    }
    if (d.z > 0){
        dsq += d.z*d.z;
    }

    return (dsq > rcrti && /* test angular criterion   */
            dmax > ((real) 1.5) * psize); /* and adjacency criterion  */
}


__device__ void cuda_tree_sumnode(int32_t body_cell_index, int32_t curr_body_index, real* mass_list, 
    cuda_vector* pos_list, real* phi_list, cuda_vector* acc_list, real eps2){

    real dr2, drab, phi_p, mr3i;
    cuda_vector dr;

    dr = pos_list[body_cell_index] - pos_list[curr_body_index];
    dr2 = dot(dr, dr);
    dr2 += eps2;
    drab = sqrtf(dr2);
    phi_p = mass_list[body_cell_index]/drab;
    mr3i = phi_p/dr2;
    phi_list[curr_body_index] -= phi_p;
    acc_list[curr_body_index] += dr * mr3i;
}





