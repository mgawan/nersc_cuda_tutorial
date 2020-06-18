/* /!\ This file must not be changed /!\ */
#define DATA_SIZE 35
#define NB_THREADS_PER_BLOCK 8

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

template <typename T>
void result_check(T*A, T*B, T*C){
    int errors = 0;
    for(int i = 0; i < DATA_SIZE; i++){
        if(A[i] + B[i] != C[i]){
            fprintf(stderr, "error in: %d / %d cpu %d gpu\n",i, A[i]+B[i], C[i]);
            errors++;
        }
    }
    if(errors == 0){
      std::cout<<"\tCorrectness Test Passed\n";
    }else{
      std::cout<<"\tCorrectness Test Failed\n";
    }
}

#define DATA_TYPE int
