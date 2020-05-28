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


#define DATA_SIZE 500000000
// kernel
__global__ 
void vector_add_kernel(const float *A, const float *B, float *C, int size){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = gridDim.x*blockDim.x;

  for(int i = idx; i < size; i+=total_threads)
    C[i] = A[i] + B[i]; // do the vector (element) add here
}

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

  cudaMalloc(&A_d, DATA_SIZE*sizeof(float));  // allocate memory for device vectors
  cudaMalloc(&B_d, DATA_SIZE*sizeof(float));  
  cudaMalloc(&C_d, DATA_SIZE*sizeof(float));  
  cudaCheck("Error in cudaMallocs"); // check if any errors during memory allocation on device
  // copy host vectors to device
  cudaMemcpy(A_d, A_h, DATA_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, DATA_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheck("host to device copy error");
  
  int blocks = 4;  // to set number of blocks
  int threads = 512; // to set number of threads per block

  //launch kernel
  vector_add_kernel<<<blocks, threads>>>(A_d, B_d, C_d, DATA_SIZE);
  cudaCheck("kernel launch error");
  // copy result vector from device to host
  cudaMemcpy(C_h, C_d, DATA_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheck("device to host copy error or kernel launch failure");

  return 0;
}

void result_check(float *A, float *B, float *C){
    int errors = 0;
    for(int i = 0; i < DATA_SIZE; i++){
        if(A[i] + B[i] != C[i]){
            errors++;
        }
    }
    std::cout<<"total errors:"<<errors<<std::endl;
}


// for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x)         // a grid-stride loop
// C[idx] = A[idx] + B[idx]; // do the vector (element) add here