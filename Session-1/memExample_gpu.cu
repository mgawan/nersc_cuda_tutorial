#include <stdio.h>
#include <assert.h>

__global__ void swap_gpu(int *a, int *b)
{
 int tmp = *a;
 *a = *b;
 *b = tmp;
}

int main()
{
 int h_a, h_b;
 h_a = 3;
 h_b = 9;

 int *dev_a, *dev_b;

 size_t varSize = sizeof(int);
 
 cudaMalloc((void **)&dev_a, varSize);
 cudaMalloc((void **)&dev_b, varSize); 
 
 cudaMemcpy(dev_a, &h_a, varSize, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, &h_b, varSize, cudaMemcpyHostToDevice); 
 
 swap_gpu<<<1,1>>>(dev_a,dev_b);
 
 cudaMemcpy(&h_a, dev_a, varSize, cudaMemcpyDeviceToHost);
 cudaMemcpy(&h_b, dev_b, varSize, cudaMemcpyDeviceToHost);

 cudaDeviceSynchronize();

 assert(h_a == 9);
 assert(h_b == 3);

 cudaFree(dev_a);
 cudaFree(dev_b);

 return 0;
}
