all: vec_add-debug-printf #stencil vec_add
stencil: stencil.cu
	nvcc --gpu-architecture=compute_70 stencil.cu -o stencil.exe

vec_add-debug-printf: vec_add-debug-printf.cu
	nvcc -G -Xcompiler -rdynamic --gpu-architecture=compute_70 $^ -o $@.exe

vec_add: vec_add.cu
	nvcc -G -Xcompiler -rdynamic --gpu-architecture=compute_70 vec_add.cu -o vec_add.exe

clean:
	rm -f stencil.exe vec_add.exe vec_add-debug-printf.exe
