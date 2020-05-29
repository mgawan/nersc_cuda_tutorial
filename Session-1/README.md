# NERSC CUDA Tutorials Session 1

## Plan
1. Say Hello to CUDA!
2. Data movement with CUDA API
3. Kernel writing and calling
4. Compiling and executing

1., 3., and 4. are covered by ```hello.c/cu```.
2. (and 3. and 4.) are covered by ```memExample_cpu.c``` and ```memExample_gpu.cu```.

## General Instructions
1. Login to cori.nersc.gov (i.e., run ```ssh myname@cori.nersc.gov``` on Linux/UNIX terminals, and Putty (most commonly used) on Windows.)
2. Then run the following commands:
   ```module purge && module load esslurm
   module load cuda```
3. Then we need to go to CoriGPU compute node, one way is to do it is to interactively logging into the CoriGPU compute node by running the following command:
   ```salloc -C gpu -N 1 -t 60 -c 10 -G 1```
4. Once you're on CoriGPU compute node, you can use a GPU. You can compile C (.c) files using "gcc" compiler, for example, and you need to compile CUDA (.cu) files using "nvcc", NVIDIA's compiler driver. For example, ```nvcc memExample_gpu.cu```.

## Resources
1. https://docs-dev.nersc.gov/cgpu/
