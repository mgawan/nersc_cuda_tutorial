# NERSC CUDA Tutorials

Load modules:
```
module load esslurm cuda
```

Start an interactive session on GPU node:
```
salloc -N 1 -C gpu --gres=gpu:1 -q interactive -t 100
```
