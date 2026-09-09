#!/usr/bin/env bash
#SBATCH --job-name=rubber-cutting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

# Invoke from the repository root, after activating its Python environment.
# Under Slurm keep the solver in the foreground so the allocation owns its lifetime.
test -f prototype/initiation_config.json || { echo 'Run from the repository root' >&2; exit 2; }
threads=${SLURM_CPUS_PER_TASK:-${THREADS:-2}}
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPU allocation' >&2; exit 2; }
output=${OUTPUT_DIR:-outputs/hpc_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}}
mkdir -p "$output"
test -z "$(ls -A "$output")" || { echo 'Output directory must be empty' >&2; exit 2; }
exec >"$output/driver.log" 2>&1
export OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is
uname -a
python --version
python -m pip freeze >"$output/packages.txt"
git rev-parse HEAD >"$output/commit.txt"
git status --short >"$output/worktree.txt"
set +e
python -u -m prototype.febio_reference --case oblique --contact-variant corrected \
    --threads "$threads" --spacing .5 --surface-spacing .0025 --depth "${DEPTH_MM:-3}" \
    --output "$output/case" --execute "$@"
result=$?
printf '%s\n' "$result" >"$output/exit_code.txt"
date -Is
exit "$result"
