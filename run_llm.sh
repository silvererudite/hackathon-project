#!/bin/bash
#SBATCH --job-name=sciops_llm
#SBATCH --account=REPLACE_WITH_FULL_ACCOUNT
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err

# BlueBEAR batch job for the SCI OPS agentic data-selection LLM task.
#
# Before first submit:
#   1. Copy config.env.example -> config.env and fill in SLURM_ACCOUNT + HF_HOME
#   2. Determine full account:  sacctmgr show assoc user=$USER format=User,Account%40,QOS%50
#   3. Fetch data if missing:    bash download_data.sh
#   4. Submit:                     sbatch run_llm.sh
#   5. Monitor:                    squeue -u $USER

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "=== sciops_llm starting $(date) ==="

if [[ -f config.env ]]; then
    # shellcheck disable=SC1091
    source config.env
fi

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in config.env (full project account name)}"
: "${HF_HOME:?Set HF_HOME in config.env (RDS project path for model weights)}"

module purge

# BlueBEAR: always load bear-apps before any 2023a application module.
module load bear-apps/2023a

# On icelake GPU nodes the CUDA Transformers module is available; on login nodes
# only the foss (CPU) build exists. Never load both PyTorch/foss and PyTorch/CUDA.
if module --ignore_cache avail Transformers/4.42.0-foss-2023a-CUDA-12.1.1 &>/dev/null; then
    echo "Loading GPU stack: Transformers/4.42.0-foss-2023a-CUDA-12.1.1"
    module load Transformers/4.42.0-foss-2023a-CUDA-12.1.1
else
    echo "Loading CPU stack: Transformers/4.42.0-foss-2023a (no GPU module on this node)"
    module load Transformers/4.42.0-foss-2023a
fi

export BLUEBEAR_LLM=1
export TRANSFORMERS_BACKEND=1
export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"

# Optional: required only for gated models (e.g. Meta Llama).
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export TRANSFORMERS_MODEL="${TRANSFORMERS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "${HF_HOME}" logs outputs

echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : $(hostname)"
echo "Account    : ${SLURM_ACCOUNT}"
echo "HF_HOME    : ${HF_HOME}"
echo "Model      : ${TRANSFORMERS_MODEL}"
echo "Python     : $(which python)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"

python task.py "$@"
