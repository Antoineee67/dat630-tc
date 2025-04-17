# Abstract

 The goal of this project was to integrate parallel computing techniques using OpenMP, MPI and CUDA to improve the performance of the Barnes-Hut treecode. 
 The resulting parallel implementations were performance tested on Chalmer’s Vera cluster to evaluate how efficient the code was on utilizing the
 parallelism on a larger system. For the pure OpenMP implementation a 5.5x
 speedup was realized. An MPI benchmark with 1 core on 4 nodes showed a
 speedup of 2.9x. For the MPI implementation in conjunction with OpenMP a
 8.8x speedup was achieved. For CUDA, there are two implementations. The
 implementation that only copies the computation tasks achieved a speedup of
 1.9x, while the implementation that copies the whole tree structure achieved
 a speedup of 11.0x.
