#!/bin/bash
srun nv-nsight-cu-cli  --section SpeedOfLight --section MemoryWorkloadAnalysis ./vec_add | grep -e Duration -e Throughput -e Correctness -e 'Hit Rate'