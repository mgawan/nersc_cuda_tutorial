NERSC CUDA Tutorials
module load esslurm cuda
salloc -N 1 -C gpu --gres=gpu:1 -q interactive -t 100
