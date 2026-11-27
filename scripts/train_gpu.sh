#!/bin/bash
# Train on every GPU of one node:
#   oarsub -l host=1,walltime=24:00 -p "gpu_count >= 2" "./scripts/train_gpu.sh vit"
#   oarsub -l host=1,walltime=24:00 -p "gpu_count >= 2" "./scripts/train_gpu.sh cnn-svm"
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL=${1:-vit}
INPUT=${2:-rgb}
torchrun --standalone --nproc_per_node=gpu -m droidhunter train \
  --data "$DH_RUNS/droidtrain/images" --model "$MODEL" --input "$INPUT" \
  --vit-source "$DH_VIT" --output "$DH_RUNS/droidtrain/models/${MODEL}_${INPUT}_seed${SEED:-42}" --seed "${SEED:-42}"
