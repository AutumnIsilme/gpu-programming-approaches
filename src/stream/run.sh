#!/bin/bash

# Generic options:

#SBATCH --account=bddur52   # Run job under project <project>
#SBATCH --time=00:01:00        # Run for a max of 1 hour

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu", "test" or "infer" partition type
#SBATCH --nodes=1           # Resources from a single node
#SBATCH --gres=gpu:1        # One GPU per node (plus 25% of node CPU and RAM per GPU)

# Run commands:
hostname
time ./stream-cuda $((64*1024*1024)) 5 256

echo "Done!"

