#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --job-name=package_data
#SBATCH --output=logs/package_data_%j.log

# Runs package_data.sh on a compute node: tars nets/ into data/*.tar.gz.part###
# chunks (<=10MB each). Pure tar/gzip/split I/O - no module load needed.
# Submit from the repo root:
#
#   mkdir -p logs
#   sbatch submit_package_data.sh                     # all archives
#   sbatch submit_package_data.sh LFC_240_ws_gt       # subset

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
bash package_data.sh "$@"
