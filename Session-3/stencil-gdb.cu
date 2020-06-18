#include "check-stencil-gdb.hpp"

// kernels
__global__
void compute_stencil_kernel(const DATA_TYPE *in, DATA_TYPE *out);
__global__
void compute_stencil_kernel_optimized(const DATA_TYPE *in, DATA_TYPE *out, size_t size);

int main(int ac, char * av[]){
  // host pointers
  DATA_TYPE *A_h, *B_h;
  // device pointers
  DATA_TYPE *A_d, *B_d;

  // allocating host arrays
  A_h = new DATA_TYPE[DATA_SIZE];
  B_h = new DATA_TYPE[DATA_SIZE];

  // initializing host vectors
  for (int i = 0; i < DATA_SIZE; i++){
    A_h[i] = i;//rand()/(DATA_TYPE)RAND_MAX;
    B_h[i] = 0;
    }

  cudaMalloc(&A_d, DATA_SIZE*sizeof(DATA_TYPE));  //allocate memory for device vectors
  cudaMalloc(&B_d, DATA_SIZE*sizeof(DATA_TYPE));
  cudaCheck("Error in cudaMallocs"); //check if any errors during memory allocation on device
  // copy host vectors to device
  cudaMemcpy(A_d, A_h, DATA_SIZE*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, DATA_SIZE*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
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
  cudaMemcpy(B_h, B_d, DATA_SIZE*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
  cudaCheck("device to host copy error or kernel launch failure");
  result_check(A_h, B_h);

  return 0;
} // end main


__global__
void compute_stencil_kernel(const DATA_TYPE *in_d, DATA_TYPE *out_d){
    int gindex = threadIdx.x + blockIdx.x * blockDim.x+RADIUS;

    //Applying stencil
    //For each thread of the block

    if(gindex ==3){
    for(int i=0;i<DATA_SIZE;i++)
        printf("in kernel A[%d/%d] = %d\n",i,DATA_SIZE,in_d[i]);
    }
    DATA_TYPE result = 0.;
    DATA_TYPE tmp;
    //Run through the radius to apply physic force (here just sum) on surrounding elements
    for(int offset = -RADIUS ; offset <= RADIUS ; offset++){
        tmp = in_d[gindex + offset];
        result += tmp;
        if(gindex ==3)
        printf("gindex(%d) offset(%d) tmp(%d) result(%d)\n",gindex,offset,tmp, result);
    }
    //Store result in global memory
    out_d[gindex] = result;
}

__global__
void compute_stencil_kernel_optimized(const DATA_TYPE *in, DATA_TYPE *out,
                            size_t size){
    __shared__ DATA_TYPE temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    //copy input array into shared memory
    temp[lindex] = in[gindex];
    if(threadIdx.x < RADIUS){
        //Copying left ghost cells
        temp[lindex - RADIUS] = 0;
        //Copying right ghost cells
        temp[lindex + BLOCK_SIZE] = 0;
    }else{
        //Copying left ghost cells
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        //Copying right ghost cells
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    //__syncthreads();// might be important ...
    //Applying stencil
    //For each thread of the block
    DATA_TYPE result = 0.;
    //Run through the radius to apply physic force (here just sum) on surrounding elements
    for(int offset = -RADIUS ; offset <= RADIUS ; offset++){
        result += temp[lindex + offset];
    }
    //Store result in global memory
    out[gindex] = result;
}
