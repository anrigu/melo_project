#!/bin/bash
#SBATCH --job-name=plot_dpr_HP_array_100_HP
#SBATCH --partition=standard
#SBATCH --time=01:00:00            # adjust wall-clock limit as needed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=wellman0
#SBATCH --output=plot_dpr_HP_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsmithl@umich.edu
#SBATCH --array=100

# ---------------------------------------------------------------------------
#  Slurm array index → holding-period value
# ---------------------------------------------------------------------------
HP=$SLURM_ARRAY_TASK_ID

echo "Plot DPR bootstrap (holding_period=$HP)"
echo "Job Id       : $SLURM_JOB_ID"
echo "Node list    : $SLURM_NODELIST"

autoload() { module load "$1" 2>/dev/null || true; }; autoload gcc

# Limit OpenBLAS / MKL threads to avoid contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --------------------------------------------------------------
#  Matplotlib cache – choose a writable directory in all cases
# --------------------------------------------------------------
if [[ -n "$SLURM_TMPDIR" ]]; then
    export MPLCONFIGDIR="$SLURM_TMPDIR/mpl_cache"
else
    export MPLCONFIGDIR="$PWD/.mpl_cache"
fi
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------------
#  Move to project root (one level up from this script)
# ---------------------------------------------------------------------------
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    cd "$SLURM_SUBMIT_DIR/.." || exit 1
else
    cd "$(dirname "$0")/.." || exit 1
fi

# Now at project root; set PYTHONPATH so bundled GameAnalysis is found
ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/marketsim/egta/gameanalysis-old:$PYTHONPATH"

# ---------------------------------------------------------------------------
#  Activate or create Python virtualenv
# ---------------------------------------------------------------------------
VENV_DIR="$HOME/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

# ---------------------------------------------------------------------------
#  Ensure SciPy is available (needed for scipy.special.comb)
# ---------------------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys
try:
    importlib.import_module('scipy')
except ModuleNotFoundError:
    print('[setup] Installing SciPy into virtualenv …')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', 'scipy'])
PY

# Give plot_bootstrap_compare access to all allocated cores
export SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-$SLURM_CPUS_PER_TASK}

echo "Using $SLURM_CPUS_ON_NODE logical cores for gap-fill parallelism"

# ---------------------------------------------------------------------------
#  Output directory for generated figures / logs
# ---------------------------------------------------------------------------
# place all outputs under a dedicated root to keep things tidy
PLOT_ROOT="plots_dpr_one_role"
PLOT_OUT="$PLOT_ROOT/holding_period_${HP}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$PLOT_OUT"

# ---------------------------------------------------------------------------
#  Run the analysis (DPR equilibrium + bootstrap)
# ---------------------------------------------------------------------------
python3 -u plot_bootstrap_compare.py \
        --hp "$HP" --dpr --fill-only 2>&1 | tee "$PLOT_OUT/plot_bootstrap_compare.log"

# ---------------------------------------------------------------------------
#  Archive gap-filled profiles for this HP inside the output dir for easy copy-back
# ---------------------------------------------------------------------------
if [[ -d gapfill_profiles/holding_period_${HP} ]]; then
    mkdir -p "$PLOT_ROOT/gapfills"
    cp -r gapfill_profiles/holding_period_${HP} "$PLOT_ROOT/gapfills/"
fi

# ---------------------------------------------------------------------------
#  Slurm efficiency report (optional)
# ---------------------------------------------------------------------------
seff "$SLURM_JOB_ID" || true 