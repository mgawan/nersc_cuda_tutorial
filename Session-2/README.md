# NERSC CUDA Tutorials Session 2

This tutorial session consists of two different parts. The first part will deal with launch configurations and go over the material studied during the presentation. The second part of this tutorial will give practical examples of memory coalescing and caching.

## Kernel Launch Configurations 
To study the launch configuration we will consider the __vector_add_kernel__ in the file __vec_add.cu__. This is a simple vector addition example where a vector __A__ and vector __B__ of same lengths are added and results are stored in a third vector __C__. The work load of this vector addition is distributed across all the available threads in all the CUDA blocks combined. For instance, if the total number of threads available is __1__ then the __for loop__ in this kernel will make total iterations equal to the length of vectors. If we increase the number of threads, the total iterations by this loop will decrease as more threads are utilized.

To get started, open the file __vec_add.cu__ and move the cursor to __line 51__