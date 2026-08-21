#!/bin/bash
# Fetch SCI OPS CSV files from the public nivlab/sciops repository.
# Run once on BlueBEAR (login node is fine -- no GPU needed).

set -euo pipefail

cd "$(dirname "$0")"

for S in 01_Original 02_Replication; do
  mkdir -p "${S}/data"
  for F in metrics scores metadata surveys items data; do
    echo "Fetching ${S}/data/${F}.csv …"
    curl -sfL "https://raw.githubusercontent.com/nivlab/sciops/main/${S}/data/${F}.csv" \
      -o "${S}/data/${F}.csv"
  done
done

echo "Done. $(find 01_Original 02_Replication -name '*.csv' | wc -l) CSV files downloaded."
