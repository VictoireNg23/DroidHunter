#!/bin/bash
# One shard of the extraction per node. Submit as an OAR array job:
#   oarsub --array 8 -l host=1,walltime=12:00 ./scripts/extract_shard.sh
# OAR_ARRAY_INDEX goes from 1 to 8; shards from 0 to 7.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
N=${DH_SHARDS:-8}
droidhunter extract --input "$DH_DATA" --output "$DH_RUNS/droidtrain/extraction" \
  --shard $((OAR_ARRAY_INDEX - 1)) --num-shards "$N" --tmp /tmp
