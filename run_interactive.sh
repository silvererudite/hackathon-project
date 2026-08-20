#!/bin/bash
# Interactive GPU session: run the agent with live output in your terminal.
# Usage:
#   bash run_interactive.sh --check-data
#   bash run_interactive.sh --generate
#   bash run_interactive.sh --probe-tools
#   bash run_interactive.sh                  # full agent (live prints)

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f config.env ]]; then
    echo "Missing config.env"
    exit 1
fi
# shellcheck disable=SC1091
source config.env

: "${SLURM_ACCOUNT:?}"
: "${HF_HOME:?}"

echo "Requesting interactive GPU (output appears in this terminal)..."
srun --account="${SLURM_ACCOUNT}" --qos=bbgpu --gres=gpu:a100:1 \
     --mem=64G --cpus-per-task=8 --time=02:00:00 --pty bash -lc "
set -euo pipefail
module purge
module load bear-apps/2023a
if module --ignore_cache avail Transformers/4.42.0-foss-2023a-CUDA-12.1.1 &>/dev/null; then
    module load Transformers/4.42.0-foss-2023a-CUDA-12.1.1
else
    module load Transformers/4.42.0-foss-2023a
fi
source '$(pwd)/config.env'
export BLUEBEAR_LLM=1 TRANSFORMERS_BACKEND=1 PYTHONUNBUFFERED=1
export HF_HOME TRANSFORMERS_CACHE=\"\${HF_HOME}/hub\" HF_HUB_CACHE=\"\${HF_HOME}/hub\"
cd '$(pwd)'
python task.py $*
"
