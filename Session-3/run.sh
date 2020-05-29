#!/bin/bash
srun nv-nsight-cu-cli  --section SpeedOfLight --section MemoryWorkloadAnalysis $1 | grep -e Duration -e Throughput -e Correctness -e 'Hit Rate'
