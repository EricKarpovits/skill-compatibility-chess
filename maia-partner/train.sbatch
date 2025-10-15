#!/bin/bash
#SBATCH --job-name=erickarp1          # Job name
#SBATCH --partition=ashton            # Partition to submit to
#SBATCH --qos=ashton                  # QoS
#SBATCH --gres=gpu:4                  # Request 4 GPUs
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=60G                     # Memory per node
#SBATCH --time=12:00:00               # Walltime (adjust as needed)
#SBATCH --output=job_%j.out           # Standard output log
#SBATCH --error=job_%j.err            # Standard error log

# Print diagnostics
pwd; hostname; date

# Load CUDA and activate your conda env
module load cuda-11.8
source /ada1/u/erickarp/miniconda3/bin/activate maia-test

# Run your program (swap this out for your actual training command)
python /ada1/u/erickarp/chess/skill-compatibility-chess/maia-partner/STT_test_tree+att.py

date
