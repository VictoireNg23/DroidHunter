#!/bin/bash
# Run on your workstation: copy models, metrics and logs (optionally images) from the cluster.
#   DH_REMOTE=user@cluster DH_RUNS=/remote/runs ./scripts/fetch_results.sh
#   ... --images   also copies the PNGs and image arrays (large)
set -euo pipefail
REMOTE=${DH_REMOTE:?Set DH_REMOTE, e.g. user@cluster}
RUNS=${DH_RUNS:?Set DH_RUNS to the runs folder on the cluster}
DEST=${DH_DEST:-./runs}
mkdir -p "$DEST"
EXCLUDE=(--exclude 'records/' --exclude '*.npz')
if [ "${1:-}" != "--images" ]; then
  EXCLUDE+=(--exclude '*.npy' --exclude 'full/')
fi
rsync -avh --partial --info=progress2 "${EXCLUDE[@]}" "$REMOTE:$RUNS/" "$DEST/"
