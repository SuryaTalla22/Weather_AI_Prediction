#!/bin/bash
#SBATCH -J flagship_atlas
#SBATCH -A m0000
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -o logs/%x-%j.out

set -euo pipefail

module load python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Update these for your environment.
export FLAGSHIP_WB2_ROOT=${FLAGSHIP_WB2_ROOT:-$PSCRATCH/data/weatherbench2}
export FLAGSHIP_OUTPUT_ROOT=${FLAGSHIP_OUTPUT_ROOT:-$PWD/outputs}

python workflows/run_dataset_audit.py
python workflows/run_truth_regimes.py
python workflows/run_deterministic_atlas.py
python workflows/run_regime_sensitivity.py
python workflows/run_growth_diagnostics.py
python workflows/run_blocking_verification.py
# Only if you have a true ensemble/member dataset configured:
python workflows/run_probabilistic_atlas.py || true
