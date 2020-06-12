#include <iostream>
#include <cstdio>
#include "check.hpp"
using namespace std;

#define RADIUS 2
#define DATA_SIZE 4 + 2*RADIUS
#define BLOCK_SIZE 4//threads per block
#define NBBLOCKS 1
void result_check(const float *A, const float *B);
// kernels
__global__
void compute_stencil_kernel(const float *in, float *out);
__global__
void compute_stencil_kernel_optimized(const float *in, float *out, size_t size);

int main(int ac, char * av[]){
  // host pointers
  float *A_h, *B_h;
  // device pointers
  float *A_d, *B_d;

  // allocating host arrays
  A_h = new float[DATA_SIZE];
  B_h = new float[DATA_SIZE];

  // initializing host vectors
  for (int i = 0; i < DATA_SIZE; i++){
    A_h[i] = i;//rand()/(float)RAND_MAX;
    B_h[i] = 0;
    }

  cudaMalloc(&A_d, DATA_SIZE*sizeof(float));  //allocate memory for device vectors
  cudaMalloc(&B_d, DATA_SIZE*sizeof(float));
  cudaCheck("Error in cudaMallocs"); //check if any errors during memory allocation on device
  // copy host vectors to device
  cudaMemcpy(A_d, A_h, DATA_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, DATA_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheck("host to device copy error");

  int blocks = NBBLOCKS;
  int threads = BLOCK_SIZE;

  for(int i=0;i<DATA_SIZE;i++)
      printf("before kernel A[%d/%d] = %f\n",i,DATA_SIZE,A_h[i]);
  //No use of shared memory
  compute_stencil_kernel<<<blocks, threads>>>(A_d, B_d);
  //compute_stencil_kernel_optimized<<<blocks, threads>>>(A_d, B_d, DATA_SIZE);
  cudaCheck("kernel launch error");
  // copy result vector from device to host
  cudaMemcpy(B_h, B_d, DATA_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheck("device to host copy error or kernel launch failure");
  result_check(A_h, B_h);

  return 0;
} // end main

void result_check(const float *A, const float *B){
    int errors = 0;
    cerr << "data size: " <<DATA_SIZE << endl;
    for(int i=0 ; i < RADIUS; i++){
        cerr << " B["<< i << "]=" << B[i] << endl;
        int j = i+NBBLOCKS*BLOCK_SIZE+RADIUS;
        cerr << " B["<< j << "]=" << B[j] << endl;
    }
    for(int i = RADIUS; i < (DATA_SIZE-RADIUS); i++){
        float tmp = 0.;
        for(int j = i-RADIUS ; j<=i+RADIUS; j++)
            tmp += A[j];
        if(true){//i<(RADIUS+2)){
            cerr << " tmp["<< i << "]=" << tmp << endl;
            cerr << " B["<< i << "]=" << B[i] << endl;
        }
        if(tmp != B[i]){
            errors++;
        }
    }
    if(errors == 0){
      std::cout<<"\tCorrectness Test Passed\n";
    }else{
      std::cout<<"\tCorrectness Test Failed("<< errors<< ")\n";
    }
}

__global__
void compute_stencil_kernel(const float *in_d, float *out_d){
    int gindex = threadIdx.x + blockIdx.x * blockDim.x+RADIUS;

    //Applying stencil
    //For each thread of the block

    if(gindex ==3){
    for(int i=0;i<DATA_SIZE;i++)
        printf("in kernel A[%d/%d] = %f\n",i,DATA_SIZE,in_d[i]);
    }
    float result = 0.;
    float tmp;
    //Run through the radius to apply physic force (here just sum) on surrounding elements
    for(int offset = -RADIUS ; offset <= RADIUS ; offset++){
        tmp = in_d[gindex + offset];
        result += tmp;
        if(gindex ==3)
        printf("gindex(%d) offset(%d) tmp(%f) result(%f)\n",gindex,offset,tmp, result);
    }
    //Store result in global memory
    out_d[gindex] = result;
}

__global__
void compute_stencil_kernel_optimized(const float *in, float *out,
                            size_t size){
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    //copy input array into shared memory
    temp[lindex] = in[gindex];
    if(threadIdx.x < RADIUS){
        //Copying left ghost cells
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        //Copying right ghost cells
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    __syncthreads();// might be important ...
    //Applying stencil
    //For each thread of the block
    float result = 0.;
    //Run through the radius to apply physic force (here just sum) on surrounding elements
    for(int offset = -RADIUS ; offset <= RADIUS ; offset++){
        result += temp[lindex + offset];
    }
    //Store result in global memory
    out[gindex] = result;
}
