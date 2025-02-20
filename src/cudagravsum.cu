
#include "treedefs.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail)
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
    if (usequad){
        while (cell_list_tail != NULL)
        {
            /* loop over node list      */
            p = &cell_list_tail->cell;
            cell_list_tail = cell_list_tail->priv;
            DOTPSUBV(dr2, dr, Pos(p), pos0); /* do mono part of force    */
            dr2 += eps2;
            drab = rsqrt(dr2);
            phi_p = Mass(p) / drab;
            mr3i = phi_p / dr2;
            DOTPMULMV(drqdr, qdr, Quad(p), dr); /* do quad part of force    */
            dr5i = ((real)1.0) / (dr2 * dr2 * drab);
            phi_q = ((real)0.5) * dr5i * drqdr;
            *phi0 -= phi_p + phi_q; /* add mono and quad pot    */
            mr3i += ((real)5.0) * phi_q / dr2;
            ADDMULVS2(acc0, dr, mr3i, qdr, -dr5i); /* add mono and quad acc    */
        }
    }                   /* if using quad moments    */
        
    /* sum cell forces w quads  */
    else /* not using quad moments   */ {
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
    }
    /* sum forces from bodies   */
    Phi(current_body) = phi0;      /* store total potential    */
    SETV(Acc(current_body), acc0); /* and total acceleration   */
    current_body->updated = TRUE;  /* mark body as done        */
    // TODO: Do we want to keep track of this? Would need synchronization logic if so.
    // nbbcalc += interact + actlen - bptr;        /* count body-body forces   */
    // nbccalc += cptr - interact;                 /* count body-cell forces   */
}

int main(int argc, char ** argv) {
	//----Problem input M x N----//
	int M = (argc >= 3) ? atoi(argv[1]) : M_;	//Read from input or default
	int N = (argc >= 3) ? atoi(argv[2]) : N_; 
	//---------------------------//


	//----Variables declaration----//
	struct timeval ts, tf;

	int i, j;

	cudaError_t err;

	//Block & Grid dimensions for the GPU
	unsigned int blockX, blockY, blockZ;
	unsigned int gridX, gridY, gridZ;
	dim3 threadsPerBlock;
	dim3 numBlock;
	
	//Matrix A, vectors b, x, to be allocated on the CPU
	float * A;
       	float * x; 
	float * b;
	//Matrix dev_A, vectors dev_b, dev_x, to be allocated on the GPU
	float * dev_A;
	float * dev_x; 
	float * dev_b;
	//Helper vector to store CPU solution for correctness checks
	float * sol;
	//-----------------------------//

	//----Query GPU properties----//
	getGPUProperties();
	//----------------------------//

	//----CPU allocations and initialization----//
	A = (float*)malloc(M * N * sizeof(float)); 	//Matrix A (size M x N)
	x = (float*)malloc(N * sizeof(float));		//Vector x (size N)
	b = (float*)malloc(M * sizeof(float));		//Vector y (size M)

	for (i = 0 ; i < M ; i++) { 				
		for (j = 0 ; j < N ; j++) 
			A[i * N + j] = (rand() % 4 + 1)*0.1;	//Initilize A
	}	
	for (i = 0 ; i < N ; i++)
		x[i] = (rand()%10 + 1) * 0.01;			//Initialize x

	sol = (float*)malloc(M * sizeof(float));
	//------------------------------------------//

	//----DGEMV A * x = b on CPU - Reference run----//
	printf("Running DGEMV with size %d x %d on the CPU - Reference version\n", M, N);
	gettimeofday(&ts, NULL);
	
	cpuDGEMV(A, x, sol, M, N);
	
	gettimeofday(&tf, NULL);
	printf("Time: %.5lf(s)\n", timetosol(ts, tf));
	//------------------------------------//


	//----GPU allocations----//
	err = cudaMalloc(&dev_A, M * N * sizeof(float));
	if (err != 0) {
		fprintf(stderr, "Error allocating matrix A on GPU\n");
		exit(-1);
	}
	err = cudaMalloc(&dev_x, N * sizeof(float));
	if (err != 0) {
		fprintf(stderr, "Error allocating vector x on GPU\n");
		exit(-1);
	}
	err = cudaMalloc(&dev_b, M * sizeof(float));
	if (err != 0) {
		fprintf(stderr, "Error allocating vector b on GPU\n");
		exit(-1);
	}
	//----------------------//

	//----Perform CPU to GPU transfers----//
	err = cudaMemcpy(dev_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) {
		fprintf(stderr, "Error copying matrix A to GPU\n");
		exit(-1);
	}

	err = cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) {
		fprintf(stderr, "Error copying vector x to GPU\n");
		exit(-1);
	}

	//------------------------------------//

	//----Shared-memory DGEMV A * x = b on GPU----//
	//TODO: Select the block dimensions (threads per block)
	//Assume a 1D block
	blockX = BLOCK_DIM;	//TODO: Select the number of threads per block
	blockY = 1;
    blockZ = 1;
	threadsPerBlock = {blockX, blockY, blockZ};

	//TODO: Select the grid dimensions (blocks)
	//Assume a 1D grid
	gridX = M;	//TODO: Select the number of blocks
	gridY = 1; 
	gridZ = 1;
	numBlock = {gridX, gridY, gridZ};
#ifdef LINEAR_REDUCTION
	printf("Running DGEMV with size %d x %d on the GPU - Shared memory version\n", M, N);
#elif BINARY_REDUCTION
	printf("Running DGEMV with size %d x %d on the GPU - Shared memory version + Binary reduction\n", M, N);
#endif
	gettimeofday(&ts, NULL);

	cudaDGEMV_shmem<<<numBlock,threadsPerBlock>>>(dev_A, dev_x, dev_b, M, N);
	

	cudaDeviceSynchronize();
	
	gettimeofday(&tf, NULL);

	//----Perform GPU to CPU transfers----//
	cudaMemcpy(b, dev_b, M * sizeof(float), cudaMemcpyDeviceToHost);
	printf("b: %1f \n", b[0]);
	//------------------------------------//

	printf("Time: %.5lf(s) -- ", timetosol(ts, tf));
	if (!checkCorrectness(sol, b, M)) 
		printf("PASS\n");


	//----Free memory on GPU and CPU----//
	cudaFree(dev_A);
	cudaFree(dev_b);
	cudaFree(dev_x);

	//Free buffers on CPU
	free(A);
	free(x);
	free(b);
	free(sol);
	//---------------------------------//

	return 0;

}
