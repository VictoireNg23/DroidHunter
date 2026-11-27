# DroidHunter

DroidHunter detects Android malware from images built out of Dalvik bytecode. Each Smali instruction becomes one RGB pixel: R encodes the opcode, G encodes the registers and B encodes the referenced method. A Vision Transformer (ViT-B/16) or a CNN feeding a linear SVM then classifies the image. The method is described in:

> V. Nganfang, S. Queyrut, Y.-D. Bromberg, V. Schiavoni, D. Mvondo, K. Tchendji Vianney. **DroidHunter: A Robust Vision-Based Detection Against Hidden Android Malware.** ACM ASIA CCS 2026. [doi:10.1145/3779208.3785386](https://doi.org/10.1145/3779208.3785386)

This repository is a complete, parallel implementation of the pipeline, from raw APKs to evaluation. It runs on a laptop for small sets and scales to clusters (multi-node extraction, multi-GPU training).

```text
 APKs ─► extract ─► records ─► split ─► vocab ─► images ─► train ─► evaluate
        apktool     per APK    80/10/10  frequency  RGB      ViT /     metrics, AUT,
        N nodes ×              stratified tables    pixels   CNN+SVM   latency
        C cores                         (train only)
```

## Highlights

- **Static only.** APKs are disassembled with apktool and never executed. All dex files are read, in deterministic multidex order.
- **Instruction-level encoding.** Each pixel stores the frequency of the opcode, the mean frequency of its registers, and the frequency of the method path (package → class → member → signature). Frequencies are learnt on the training split only, then min-max scaled to 0–255.
- **Optional 3-gram channels.** Three extra channels, aligned pixel for pixel, encode the 3-gram ending at each instruction (`--input hybrid`, 6 channels).
- **Fast at scale.**
  - Extraction runs one apktool per core and can be sharded across nodes, and an interrupted job resumes where it stopped.
  - Images are resized once and stored as memory-mapped arrays.
  - Training uses `torchrun` (DDP) with mixed precision and resumes from the last epoch.
- **Leakage checks.** SHA-256 deduplication, conflicting labels rejected, and the vocabulary fitted on the training split only. External test sets are refused if they overlap the training data.
- **Full reporting.** Every run writes accuracy, precision, recall, F1, MCC, ROC-AUC, per-year results with AUT, latency at batch size 1, and resource usage, as CSV and JSON.

## Datasets

The APK datasets are not distributed with this repository because of their size. Samples can be downloaded from [AndroZoo](https://androzoo.uni.lu) by SHA-256, after requesting an API key.

| Dataset | Content | Malware | Benign |
| --- | --- | --- | --- |
| DroidTrain | Drebin, Contagio, AndroZoo (2011–2018) | 28,843 | 137,703 |
| DroidObPack 1 | DroidTrain malware + light repackaging | 24,196 | 0 |
| DroidObPack 2 | DroidTrain malware + code obfuscation | 24,196 | 0 |
| DroidDrift | AndroZoo (2019–2024) | 42,477 | 219,522 |

Any folder containing `Benign/` and `Malware/` works as input.

## Installation

Requires Python ≥ 3.10, Java and [apktool](https://apktool.org).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA build
pip install -e .
python -m unittest discover -s tests
```

On a cluster, `scripts/install.sh` installs everything in a shared folder: Miniconda, Java, apktool and the ViT weights. Each job then only runs `source scripts/env.sh` (see [Cluster usage](#cluster-usage)).

## Quick start

The dataset folder contains `Benign/` and `Malware/`, with APKs at any depth.

```bash
droidhunter extract --input data/D0 --output runs/d0/extraction
droidhunter merge   --output runs/d0/extraction
droidhunter split   --manifest runs/d0/extraction/manifest.csv --output runs/d0/splits.json
droidhunter vocab   --manifest runs/d0/extraction/manifest.csv --splits runs/d0/splits.json --output runs/d0/vocab
droidhunter images  --manifest runs/d0/extraction/manifest.csv --splits runs/d0/splits.json \
                    --vocab runs/d0/vocab --output runs/d0/images
droidhunter train   --data runs/d0/images --model vit     --output runs/d0/models/vit
droidhunter train   --data runs/d0/images --model cnn-svm --output runs/d0/models/cnn_svm
```

Evaluate on an external set (e.g. obfuscated malware) with the training vocabulary:

```bash
droidhunter extract --malware data/DroidObPack1 --output runs/obpack1/extraction
droidhunter merge   --output runs/obpack1/extraction
droidhunter images  --manifest runs/obpack1/extraction/manifest.csv --vocab runs/d0/vocab \
                    --exclude-splits runs/d0/splits.json --name droidobpack1 --output runs/external
droidhunter evaluate --model-dir runs/d0/models/vit --data runs/external/droidobpack1 --output runs/eval/vit_obpack1
```

For temporal drift, pass `--year-regex '(20[12][0-9])'` to `extract`, so that the year is read from the APK path. `evaluate` then reports metrics per year and the Area Under Time (AUT).

Each command has `--help`. `droidhunter report` compares results with the figures reported in the paper.

## Commands

| Command | Purpose | Main options |
| --- | --- | --- |
| `extract` | APK → per-APK instruction records | `--jobs`, `--shard/--num-shards`, `--tmp`, `--year-regex` |
| `merge` | Merge shard manifests, deduplicate | |
| `split` | Stratified 80/10/10 partition | `--seed`, `--max-instructions` |
| `vocab` | Frequency tables on the training split | `--scale linear\|log\|rank`, `--trigrams`, `--path-depth` |
| `images` | Records → PNGs + training arrays | `--features rgb tri`, `--sizes 224 400`, `--no-png`, `--keep-records` |
| `train` | ViT or CNN+SVM | `--input rgb\|trigram\|hybrid`, `--mask`, `--epochs`, `--amp`, `--scheduler` |
| `evaluate` | Metrics, per-year AUT, latency | `--latency N`, `--torch-model` |
| `ablation` | Retrain with channels disabled, one variant per GPU | `--gpus 0,1,2`, `--eval` |
| `report` | Table against the paper's figures | |

## Outputs

| File | Content |
| --- | --- |
| `extraction/manifest.shardNNN.csv` | per APK: status, instruction count, extraction time, error |
| `images/<split>/full/rgb/<sha256>.png` | full-size image of each APK |
| `images/<split>/rgb_<size>.npy` | resized images, memory-mapped, used for training |
| `models/<name>/metrics_train.csv` | per epoch: loss, validation accuracy / precision / recall / F1, confusion counts, LR, throughput |
| `models/<name>/metrics_test.csv` | final test metrics (CNN head and SVM for `cnn-svm`) |
| `eval/<name>/metrics.csv`, `per_year.csv`, `latency.csv` | external evaluation, per-year metrics with AUT, latency at batch size 1 (ms) |
| `resources.csv` (each stage) | wall time, CPU time, average cores, peak RAM, peak GPU memory |

CSV files are appended while a job runs, so you can read them during long runs.

## Channel ablation

```bash
droidhunter ablation --data runs/d0/images --model vit --gpus 0,1,2 \
  --eval runs/external/droidobpack1 --output runs/ablation/vit
```

The command trains `full` (RGB), `no_R` (no opcode), `no_G` (no registers) and `no_B` (no method) on the same splits. A disabled channel is set to zero, as in §5.6 of the paper. `comparison.json` reports the accuracy drop of each variant against `full`.

## Cluster usage

The scripts in `scripts/` target OAR (Grid'5000) but only use plain shell, so they are easy to adapt to Slurm.

```bash
export DH_BASE=/path/to/shared/workdir        # code, datasets, runs and tools live here
bash scripts/install.sh                        # once, on one machine
source scripts/env.sh                          # in every job

oarsub --array 8 -l host=1,walltime=12:00 ./scripts/extract_shard.sh    # 8 nodes, one shard each
droidhunter merge --output $DH_RUNS/droidtrain/extraction
oarsub -l host=1,walltime=24:00 -p "gpu_count >= 2" "./scripts/train_gpu.sh vit rgb"
DH_REMOTE=user@cluster DH_RUNS=/remote/runs ./scripts/fetch_results.sh  # from your workstation
```

Expect about 10–20 s per APK and per core for extraction. For about 166k APKs, the disk needs about 25 GB per array at 224×224 and 80 GB at 400×400, plus the PNGs.

## Design choices

- **Operands.** op1 is the set of registers of the instruction, and op2 is the first non-register operand. Strings, labels and literals map to `<STR>`, `<LABEL>` and `<LIT>`, so the vocabulary stays bounded.
- **Method tree.** The value of op2 is the mean frequency of its path prefixes, up to `--path-depth` levels, after which min-max scaling is applied.
- **Unseen tokens.** A token never seen in training gets intensity 0, the same as the padding.
- **Image layout.** Pixels follow program order on the smallest square that holds them, zero padded. Images are resized to 224×224 for the ViT and 400×400 for the CNN.
- **CNN + SVM.** The CNN is trained with cross-entropy. The best checkpoint on validation then provides the features, and a linear SVM (C = 0.2) is fitted on the training features. `--svm svc` uses `SVC(kernel="linear")`; the default `LinearSVC` fits the same model family much faster.

## Project layout

```text
droidhunter/
  smali.py     Smali parsing: instruction -> (opcode, registers, operand)
  extract.py   parallel, sharded, resumable apktool extraction
  records.py   compact per-APK instruction storage (.npz)
  splits.py    stratified partitions
  vocab.py     frequency tables, 3-gram hashing, intensity scaling
  images.py    pixel encoding, PNGs, memory-mapped arrays
  data.py      PyTorch dataset, normalisation, channel masks
  models.py    ViT-B/16 (3 or 6 channels), CNNs
  train.py     training loop (DDP, AMP, resume), SVM
  evaluate.py  metrics, per-year AUT, latency
  ablation.py  channel ablation scheduler
  report.py    comparison with the paper
tests/         unit tests
scripts/       install and cluster job scripts
```

## Citation

```bibtex
@inproceedings{nganfang2026droidhunter,
  title     = {DroidHunter: A Robust Vision-Based Detection Against Hidden Android Malware},
  author    = {Nganfang, Victoire and Queyrut, Simon and Bromberg, Y{\'e}rom-David and Schiavoni, Valerio and Mvondo, Djob and Tchendji Vianney, Kengne},
  booktitle = {Proceedings of the ACM Asia Conference on Computer and Communications Security (ASIA CCS '26)},
  year      = {2026},
  doi       = {10.1145/3779208.3785386}
}
```
