# NERSC CUDA Tutorials Session 3

This tutorial session consists of two different parts. The first part will deal with debugging thanks to printf statement in a very simple CUDA kernel (not many threads). The second part of this tutorial consists in using cuda-memcheck to detect an error in dealing with read/write in arrays, due to wrong usage of cuda threads identification.

## Compiling Kernels
After connecting to a gpu in interactive just use the `make` command.

## Kernel Launch Configurations 
To run the printf version: `srun ./vec_add-debug-printf.exe`
To run the cuda-memcheck version: `srun cuda-memcheck ./vec_add-debug-memcheck.exe`
To run the cuda-gdb version: `srun --pty cuda-gdb ./vec_add-debug-memcheck.exe`

## Debug with printf
Running the program you should observe:

```
[...]
idx(...) total_threads(...) strides_per_threads(...)
before i(31) A(31) B(31) C(0)
after i(31) A(31) B(31) C(62)
error in: 32 / 64 cpu 0 gpu
error in: 33 / 66 cpu 0 gpu
error in: 34 / 68 cpu 0 gpu
	Correctness Test Failed
```

Your mission, if you accept it, it is to get the correct answer during the check function.
Obtaining:

```
[...]
idx(...) total_threads(...) strides_per_threads(...)
before i(31) A(31) B(31) C(0)
after i(31) A(31) B(31) C(62)
	Correctness Test Passed
```

Feel free to add more printf inside the code.

## Debug with cuda-memcheck
Sometimes (often?), printf is not enough, because there are too many cuda threads and you don't know what to print for example.
A quick and powerfull sanity check can be made with some of cuda debug tools, as cuda-memcheck.

First try to run the code version **without** cuda-memcheck: `srun ./vec_add-debug-memcheck.exe`
It seems like everything is executing correctly...

To run the cuda-memcheck version: `srun cuda-memcheck ./vec_add-debug-memcheck.exe`
You should obtain something like:

```
========= CUDA-MEMCHECK
========= Invalid __global__ read of size 4
=========     at 0x00000e20 in /global/u1/h/hbrunie/nersc_cuda_tutorial/Session-3/vec_add-debug-memcheck.cu:66:vector_add_kernel_memory(int const *, int const *, int*, int, int)
=========     by thread (7,0,0) in block (3,0,0)
=========     Address 0x2aaae24000fc is out of bounds
=========     Device Frame:/global/u1/h/hbrunie/nersc_cuda_tutorial/Session-3/vec_add-debug-memcheck.cu:66:vector_add_kernel_memory(int const *, int const *, int*, int, int) (vector_add_kernel_memory(int const *, int const *, int*, int, int) : 0xe20)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/usr/lib64/libcuda.so.1 (cuLaunchKernel + 0x346) [0x297db6]
=========     Host Frame:./vec_add-debug-memcheck.exe [0x165e9]
=========     Host Frame:./vec_add-debug-memcheck.exe [0x16677]
=========     Host Frame:./vec_add-debug-memcheck.exe [0x4c9d5]
=========     Host Frame:./vec_add-debug-memcheck.exe [0x4713]
=========     Host Frame:./vec_add-debug-memcheck.exe (_Z52__device_stub__Z24vector_add_kernel_memoryPKiS0_PiiiPKiS0_Piii + 0x148) [0x45af]
=========     Host Frame:./vec_add-debug-memcheck.exe (_Z24vector_add_kernel_memoryPKiS0_Piii + 0x38) [0x45f7]
=========     Host Frame:./vec_add-debug-memcheck.exe (main + 0x263) [0x42f5]
=========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xea) [0x20f8a]
=========     Host Frame:./vec_add-debug-memcheck.exe (_start + 0x2a) [0x3e9a]
=========

[...]
```

It seems like our kernel is reading at the wrong place for certain threads.
It also gives the line inside the code: line 66.
It also gives the identity of the threads who did wrong: **thread (7,0,0) in block (3,0,0)**.
Now, you can use gdb to investigate: good luck!
