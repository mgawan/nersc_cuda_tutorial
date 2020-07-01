# NERSC CUDA Tutorials Session 4

This tutorial introduces basic commands to generate profiler reports using the NVidia tools `NSight Sytems` and `NSight Compute`.  As a trial kernel, we take the vector addition example for Tutorial Session 2.  When generating the profiler reports on remote systems via SSH (perhaps through multiple firewalls), a convenient workflow could be to:

- generate the reports on the remote system using the profiling tool's respective command line interface, then transfer reports to a local machine for analysis with the visual profiling interface,

or alternatively, 

- access the remote system and use the visual profiling interfaces via a remote desktop. 

For complete information, refer to the official documentation: 
- https://docs.nvidia.com/nsight-compute/index.html
- https://docs.nvidia.com/nsight-systems/index.html

## Get the test kernel
From the `/nersc_cuda_tutorial` directory, (and assuming the `vec_add` executable  from Session-2 has already been compiled), navigate to the Session-4 directory and make a copy of the executable:
```
cd Session-4
cp ../Session-2/vec_add
```

## NSight Systems
To generate an NSight Systems report, we use the `profile` command of nsys:

```
srun nsys profile -o "nsys_profile" -f true  --stats=true ./vec_add
```

This generates a report `nsys_profile.qdrep` (as well as an SQLite database `nsys_profile.sqlite`, which contains all the profiling information from the run).  The command line arguments here tell us:
- `-o nsys_profile`: generate a report with name 'nsys_profile'.
- `-f true`: overwrite possible duplicates with name 'nsys_profile'.
- `--stats=true`: output profiling statistics to the command line; when set to true, also generates an SQLite database with the collected profiling information.

Some additional command line arguments you may find useful:
- `-y #`: delay data collection by `#` seconds.
- `-d #`: collect data for `#` seconds.

For more information, consult the official NSight Systems documentation (listed above).  You can open a report with:

```
nsight-sys nsys_profile.qdrep
```

## NSight Compute
To generate an NSight Compute report,  you may use the command line interface to NSight Compute, `nv-nsight-cu-cli`:

```
srun nv-nsight-cu-cli -o cu_profile  -f -k "vector_add_kernel" ./vec_add
```

This generates a report `cu_profile.nsight-cuprof-report`.  The command line arguments here tell us:
- `-o cu_profile`: generate a report with name 'cu_profile'.
- `-f`: overwrite possible duplicates with name 'cu_profile'.
- `-k "vector_add_kernel"`: target kernel for which to generate report is named `vector_add_kernel`.

Additional command line arguments you may find useful when profiling more complicated kernels:
- `-s #`: skip `#` kernels before collecting data.
- `-c #`: collect data for only `#`kernel launches.

As before, consult the official NSight Compute documentation for complete information.  You can open a report with:

```
nv-nsight-cu cu_profile.nsight-cuprof-report
```
