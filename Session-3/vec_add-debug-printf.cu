#include <iostream>
#include "check-debug-printf.hpp"

/* /!\ You might have to change the value of this constant ... /!\ */
#define NBBLOCKS 4

// kernel
__global__
void vector_add_kernel_memory(const DATA_TYPE *A, const DATA_TYPE *B, DATA_TYPE *C, int size, int stride_size);

int main(){

  DATA_TYPE *A_h, *B_h, *C_h; // host pointers
  DATA_TYPE *A_d, *B_d, *C_d;// device pointers

  A_h = new DATA_TYPE[DATA_SIZE];  // allocating host arrays
  B_h = new DATA_TYPE[DATA_SIZE];
  C_h = new DATA_TYPE[DATA_SIZE];

  for (int i = 0; i < DATA_SIZE; i++){  // initializing host arrays
    A_h[i] = i;//rand()/(DATA_TYPE)RAND_MAX;
    B_h[i] = i;//rand()/(DATA_TYPE)RAND_MAX;
    C_h[i] = 0;
    }

  cudaMalloc(&A_d, DATA_SIZE*sizeof(DATA_TYPE));  //allocate memory for device arrays
  cudaMalloc(&B_d, DATA_SIZE*sizeof(DATA_TYPE));
  cudaMalloc(&C_d, DATA_SIZE*sizeof(DATA_TYPE));
  cudaCheck("Error in cudaMallocs"); //check if any errors during memory allocation on device
  // copy host arrays to device
  cudaMemcpy(A_d, A_h, DATA_SIZE*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, DATA_SIZE*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaCheck("host to device copy error");

  int blocks = NBBLOCKS;  // to set number of blocks
  int threads = NB_THREADS_PER_BLOCK; // to set number of threads per block

  //uncomment the below kernel for studying memory caching
  vector_add_kernel_memory<<<blocks, threads>>>(A_d, B_d, C_d, DATA_SIZE,1);
  cudaCheck("kernel launch error");
  // copy result vector from device to host
  cudaMemcpy(C_h, C_d, DATA_SIZE*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
  cudaCheck("device to host copy error or kernel launch failure");

  result_check(A_h, B_h, C_h);

  return 0;
} // end main


// kernel for stuyding effect of caching
__global__
void vector_add_kernel_memory(const DATA_TYPE *A, const DATA_TYPE *B, DATA_TYPE *C, int size, int stride_size){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = gridDim.x*blockDim.x;
  /* /!\ You might have to change the expression of strides_per_thread ... /!\ */
  int strides_per_thread = size/(total_threads*stride_size);

  if(idx==0)
    printf("total_threads*stride_size modulo size --> %d\n",size % total_threads*stride_size);
  printf("idx(%d) total_threads(%d) strides_per_threads(%d) \n",idx,total_threads,strides_per_thread);
  for(int j = 0; j < strides_per_thread; j++){
      int stride_begin = stride_size * idx + j * stride_size * total_threads;
      int stride_end = stride_size + stride_begin;

      for(int i = stride_begin; i < stride_end; i++ ){
        if(idx ==31)
            printf("before i(%d) A(%d) B(%d) C(%d)\n",i,A[i],B[i],C[i]);
        C[i] = A[i] + B[i];
        if(idx ==31)
            printf("after i(%d) A(%d) B(%d) C(%d)\n",i,A[i],B[i],C[i]);
      }
  }
}
