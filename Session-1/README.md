# NERSC CUDA Tutorials Session 1

## Agenda
1. Say Hello to CUDA!
2. Writing a simple kernel and invoking one.
3. Compiling and executing your first CUDA program.
4. Data movement with CUDA API.

Points 1., 2., and 3. are covered via ```hello.cu``` program. Point 4. (and, of course, with the inclusion of concepts learned from points before it) is covered by ```memExample_gpu.cu```. Both examples have a corresponding C version for comparison. 

## General Instructions
1. Login to ```cori.nersc.gov``` (i.e., run ```ssh yourusername@cori.nersc.gov``` on Linux/UNIX terminals, and Putty (most commonly used) on Windows.)
2. Then, run the following commands:
   ```
   module purge && module load esslurm
   module load cuda
   ```
3. Next, we need to go to CoriGPU compute node, one way to do it is to interactively log into the CoriGPU compute node by running the following command:
   ```
   salloc -C gpu -N 1 -t 60 -c 10 -G 1
   ```
4. Finally, once you're on CoriGPU compute node, you can use a GPU. You can compile C (.c) files using "gcc" compiler, for example, and you need to compile CUDA (.cu) files using "nvcc", NVIDIA's compiler driver. For example, 
   ```
   nvcc memExample_gpu.cu
   srun -n1 ./a.out
   ```

## Helpful Resources
1. https://docs-dev.nersc.gov/cgpu/
