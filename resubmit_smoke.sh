#!/bin/bash
# Fix Windows line endings and submit smoke test. Run on BlueBEAR login node.
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' submit.sh run_llm.sh download_data.sh config.env 2>/dev/null || true
bash submit.sh --smoke-test
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Then:          tail -f logs/llm_*.out"
