all: stencil vec_add
stencil: stencil.cu
	nvcc --gpu-architecture=compute_70 stencil.cu -o stencil.exe

vec_add: vec_add.cu
	nvcc --gpu-architecture=compute_70 vec_add.cu -o vec_add.exe

clean:
	rm stencil.exe vec_add.exe
