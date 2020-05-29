#include <iostream>

//macro for checking errors thrown by CUDA API from https://github.com/olcf/cuda-training-series
#define cudaCheck(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


#define DATA_SIZE 1024*1024*32
#define NBBLOCKS 2*1024*32
#define BLOCK_SIZE 512
#define STRIDE 1
#define DATA_PER_BLOCK BLOCK_SIZE * STRIDE
// kernel
void result_check(float *A, float *B, float *C);
__global__ 
void vector_add_kernel_memory(const float *A, const float *B, float *C, int size, int stride_size);

int main(){

  float *A_h, *B_h, *C_h; // host pointers
  float *A_d, *B_d, *C_d;// device pointers

  A_h = new float[DATA_SIZE];  // allocating host arrays
  B_h = new float[DATA_SIZE];
  C_h = new float[DATA_SIZE];

  for (int i = 0; i < DATA_SIZE; i++){  // initializing host vectors
    A_h[i] = rand()/(float)RAND_MAX;
    B_h[i] = rand()/(float)RAND_MAX;
    C_h[i] = 0;
    }

  cudaMalloc(&A_d, DATA_SIZE*sizeof(float));  //allocate memory for device vectors
  cudaMalloc(&B_d, DATA_SIZE*sizeof(float));  
  cudaMalloc(&C_d, DATA_SIZE*sizeof(float));  
  cudaCheck("Error in cudaMallocs"); //check if any errors during memory allocation on device
  // copy host vectors to device
  cudaMemcpy(A_d, A_h, DATA_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, DATA_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheck("host to device copy error");
  
  int blocks = NBBLOCKS;  // to set number of blocks
  int threads = BLOCK_SIZE; // to set number of threads per block

  //uncomment the below kernel for studying memory caching
  vector_add_kernel_memory<<<blocks, threads>>>(A_d, B_d, C_d, DATA_SIZE, STRIDE);
  cudaCheck("kernel launch error");
  // copy result vector from device to host
  cudaMemcpy(C_h, C_d, DATA_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheck("device to host copy error or kernel launch failure");

  result_check(A_h, B_h, C_h);

  return 0;
} // end main

void result_check(float *A, float *B, float *C){
    int errors = 0;
    for(int i = 0; i < DATA_SIZE; i++){
        if(A[i] + B[i] != C[i]){
            errors++;
        }
    }
    if(errors == 0){
      std::cout<<"\tCorrectness Test Passed\n";
    }else{
      std::cout<<"\tCorrectness Test Failed\n";
    }
} 

// kernel for stuyding effect of caching
__global__ 
void vector_add_kernel_memory(const float *A, const float *B, float *C, int size, int stride_size){
  int globidx = threadIdx.x + blockIdx.x * blockDim.x;
  int locidx = threadIdx.x;
  __shared__ float tempA[DATA_PER_BLOCK];
  __shared__ float tempB[DATA_PER_BLOCK];
  __shared__ float tempC[DATA_PER_BLOCK];
  for(int i=0; i<stride_size; i++){
    tempA[locidx*stride_size+i] = A[globidx*stride_size+i];
    tempB[locidx*stride_size+i] = B[globidx*stride_size+i];
  }

  for(int i=0; i<stride_size; i++){
        tempC[locidx*stride_size+i] = tempA[locidx*stride_size+i] + tempB[locidx*stride_size+i];
  }

  for(int i=0; i<stride_size; i++)
       C[globidx*stride_size+i] = tempC[locidx*stride_size+i];
  //for(int j = globidx*size/total_threads; j < (idx*size/total_threads+ size/total_threads); j++)
  //    C[j] = tempC[j];
}
