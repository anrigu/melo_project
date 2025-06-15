#!/bin/bash
#SBATCH --job-name=mobi_zi_egta
#SBATCH --partition=standard
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5g
#SBATCH --gres=gpu:1               # comment-out if CPU-only
#SBATCH --account=wellman98

# ---------- job array: one holding-period per task ----------
# edit the list below and set the --array range accordingly
HOLDING_PERIODS=(30 50 70 100)
# To launch, run:  sbatch --array=0-$(( ${#HOLDING_PERIODS[@]} - 1 )) $0

HP=${HOLDING_PERIODS[$SLURM_ARRAY_TASK_ID]}

echo "Running EGTA for holding_period=$HP  (task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT)"

echo "Job Id              : $SLURM_JOB_ID"
echo "Node list           : $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module load cuda cudnn    # adjust versions if needed
module load gcc           # Great Lakes provides GCC via modules

# --------- Python environment ---------
source $HOME/venv/bin/activate   # update path to your venv

# --------- Run experiment ---------
python3 -u examples/run_mobi_zi_role_symmetric_analysis.py --holding-period $HP --output-root results/rsg_mobi_zi_egta_batch 