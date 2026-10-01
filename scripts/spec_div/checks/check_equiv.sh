#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=mhk_equiv
#SBATCH --mem=4G
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/check_mhk_equiv_%j.out

# Gate on launching the generator array: asserts spectral_divergence_gen.mhk_reference
# is edge-for-edge identical to the fixed net_functions.mhk_network. Submit from the
# repo root, pointing NET_FUNCTIONS_DIR at the companion repository's scripts/:
#
#     mkdir -p logs
#     NET_FUNCTIONS_DIR=<path-to-Scale-Free-Social-Contagion>/scripts \
#         sbatch scripts/spec_div/checks/check_equiv.sh
#     grep EQUIV_CHECK_RESULT logs/check_mhk_equiv_<jobid>.out    # want PASS

source <path-to-venv>/bin/activate
module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/spec_div/checks/check_mhk_equiv.py
