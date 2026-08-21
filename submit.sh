#!/bin/bash
# Submit run_llm.sh with account/HF paths from config.env.
#
#   cp config.env.example config.env   # edit SLURM_ACCOUNT and HF_HOME
#   bash submit.sh
#   bash submit.sh --smoke-test        # pass args through to task.py

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f config.env ]]; then
    echo "Missing config.env — copy config.env.example and edit it first."
    exit 1
fi

# shellcheck disable=SC1091
source config.env

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in config.env}"
: "${HF_HOME:?Set HF_HOME in config.env}"

mkdir -p logs outputs

exec sbatch --account="${SLURM_ACCOUNT}" run_llm.sh "$@"
