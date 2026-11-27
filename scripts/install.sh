#!/bin/bash
# One-time install in $DH_TOOLS: Miniconda, Python env, Java, apktool and the ViT weights.
# Run it once on one machine; every machine sharing $DH_BASE then reuses it via env.sh.
#   DH_BASE=/path/to/workdir bash scripts/install.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$DH_TOOLS/bin" "$DH_TOOLS/models"
# The tools are shared by every machine: only one install may run at a time.
exec 9>"$DH_TOOLS/.install.lock"
flock -n 9 || { echo "Another machine is already installing; wait for it, then just source env.sh." >&2; exit 1; }

if [ ! -x "$DH_TOOLS/miniconda3/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$DH_TOOLS/miniconda3"
fi
source "$DH_TOOLS/miniconda3/bin/activate"
conda env list | grep -q '^droidhunter ' || conda create -n droidhunter --override-channels -c conda-forge python=3.10 openjdk=17 -y
conda activate droidhunter
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e "$DH_CODE" "huggingface_hub[cli]"

if [ ! -f "$DH_TOOLS/bin/apktool.jar" ]; then
  wget -q https://github.com/iBotPeaches/Apktool/releases/download/v2.9.3/apktool_2.9.3.jar -O "$DH_TOOLS/bin/apktool.jar"
fi
cat > "$DH_TOOLS/bin/apktool" <<EOF
#!/bin/sh
exec java \$_JAVA_OPTIONS -jar $DH_TOOLS/bin/apktool.jar "\$@"
EOF
chmod +x "$DH_TOOLS/bin/apktool"

if [ ! -f "$DH_VIT/config.json" ]; then
  huggingface-cli download google/vit-base-patch16-224-in21k --local-dir "$DH_VIT"
fi

apktool --version
python -c "import torch, droidhunter; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Install OK. On each machine: source $DH_CODE/scripts/env.sh"
