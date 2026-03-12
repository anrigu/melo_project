#!/bin/bash
#SBATCH --job-name=mobi_zi_test
#SBATCH --partition=standard
#SBATCH --time=10:00:00           # 10 h should finish a tiny pilot run
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4g
#SBATCH --account=wellman98

# Optional GPU -- comment out if not required
##SBATCH --gres=gpu:1

HP=50   # single holding-period value to profile

echo "Pilot EGTA run (holding_period=$HP)"

echo "Job Id              : $SLURM_JOB_ID"
echo "Node list           : $SLURM_NODELIST"

module load cuda cudnn            # remove if CPU-only
module load gcc

# Activate your Python environment
source $HOME/venv/bin/activate

# Write output to a scratch sub-folder so prod results stay clean
OUT_DIR="results/pilot_egta_$SLURM_JOB_ID"

python3 -u examples/run_mobi_zi_role_symmetric_analysis.py \
        --holding-period $HP \
        --output-root $OUT_DIR

# BASIC RESOURCE USAGE SUMMARY
seff $SLURM_JOB_ID || true 