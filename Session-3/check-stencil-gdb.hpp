/* /!\ This file must not be changed /!\ */

#include <cstdio>
#include <iostream>
using namespace std;
#define DATA_TYPE int
#define RADIUS 2
#define DATA_SIZE (4 + 2*RADIUS)
#define BLOCK_SIZE 4//threads per block
#define NBBLOCKS 1
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
// control verbose printf statements 
#define verbprintf(LEVEL, ...)                                                           \     if(verbose_level >= LEVEL)                                                           \         fprintf(stderr, __VA_ARGS__);

void result_check(const DATA_TYPE *A, const DATA_TYPE *B){
    int errors = 0;
    cerr << "data size: " <<DATA_SIZE << endl;
    for(int i=0 ; i < RADIUS; i++){
        cerr << " B["<< i << "]=" << B[i] << endl;
        int j = i+NBBLOCKS*BLOCK_SIZE+RADIUS;
        cerr << " B["<< j << "]=" << B[j] << endl;
    }
    for(int i = RADIUS; i < (DATA_SIZE-RADIUS); i++){
        DATA_TYPE tmp = 0.;
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
