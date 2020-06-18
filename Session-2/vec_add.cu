#include <iostream>

//some of the materials here was reused or based on NERSC/OLCF cuda training-series (https://github.com/olcf/cuda-training-series)
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

void result_check(float *A, float *B, float *C);

// kernel
__global__ 
void vector_add_kernel(const float *A, const float *B, float *C, int size);
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
  
  int blocks = 1;  // to set number of blocks
  int threads = 512; // to set number of threads per block
  // set memory stride size
  
  //*************************************************************************************///
  //*********** uncomment the below kernel for studying launch configurations ***********///
  //*************************************************************************************///

  vector_add_kernel<<<blocks, threads>>>(A_d, B_d, C_d, DATA_SIZE);

  //*************************************************************************************///
  //*********** uncomment the below two lines to study memory caching and coalescing ***********///
  //*************************************************************************************///
  //  int mem_stride = 1; 
  //  vector_add_kernel_memory<<<blocks, threads>>>(A_d, B_d, C_d, DATA_SIZE, mem_stride);

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

// strided kernel for launch configurations
__global__ 
void vector_add_kernel(const float *A, const float *B, float *C, int size){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = gridDim.x*blockDim.x;

  for(int i = idx; i < size; i+=total_threads)
    C[i] = A[i] + B[i]; 
}

// kernel for stuyding effect of caching
__global__ 
void vector_add_kernel_memory(const float *A, const float *B, float *C, int size, int stride_size){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = gridDim.x*blockDim.x;
  int strides_per_thread = size/(total_threads*stride_size);

  for(int j = 0; j < strides_per_thread; j++){
      int stride_begin = stride_size * idx + j * stride_size * total_threads;
      int stride_end = stride_size + stride_begin;

      for(int i = stride_begin; i < stride_end; i++ ){
        C[i] = A[i] + B[i];
      }
  }
}
