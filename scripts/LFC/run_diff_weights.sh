#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00 # calibrated: measured 355-363s across k=2/16/32 (N=240 solve cost dominates, not k), 3x margin
#SBATCH --output=logs/%x_%A_%a.out
# Mail intentionally left disabled by default (see submit_diff_weights.sh comment
# for the total task count this drives).
# #SBATCH --mail-user=<your-email>
# #SBATCH --mail-type=FAIL

set -euo pipefail

module load StdEnv/2020 gcc/9.3.0 graph-tool/2.56 python/3.10

MODEL_TYPE=${MODEL_TYPE:-ws}
N_P=100

if [ "$SEED" -eq 1 ]; then
    # seed 1 sweeps the full k range (16 values) x full p sweep (100 values)
    K_VALUES=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32)
    P_IDX=$(( SLURM_ARRAY_TASK_ID % N_P ))
    K_IDX=$(( SLURM_ARRAY_TASK_ID / N_P ))
    K=${K_VALUES[$K_IDX]}
else
    # seeds 2-100: k=16 only, full p sweep
    K=16
    P_IDX=$SLURM_ARRAY_TASK_ID
fi

P=$(python3 -c "import numpy as np; print(np.linspace(0.001, 1, $N_P)[$P_IDX])")

echo "seed=$SEED model_type=$MODEL_TYPE k=$K p=$P (array task $SLURM_ARRAY_TASK_ID)"

python scripts/LFC/LFC_net_gen_diff_weights.py --seed "$SEED" --model_type "$MODEL_TYPE" --k "$K" --p "$P"
