# Shared paths for every job. Set DH_BASE once (e.g. in ~/.bashrc) to your working directory:
#   export DH_BASE=/srv/storage/<group>/datasets/DroidHunter
#   source $DH_BASE/DroidHunter_Code/scripts/env.sh
: "${DH_BASE:?Set DH_BASE to the folder holding the code, the datasets and the runs}"
export DH_CODE=${DH_CODE:-$DH_BASE/DroidHunter_Code}
export DH_DATA=${DH_DATA:-$DH_BASE/Dataset/Dataset_D0}
export DH_RUNS=${DH_RUNS:-$DH_BASE/runs}
export DH_TOOLS=${DH_TOOLS:-$DH_BASE/tools}     # miniconda3/, bin/apktool, models/
export DH_VIT=${DH_VIT:-$DH_TOOLS/models/vit-base-patch16-224-in21k}
export PATH=$DH_TOOLS/bin:$PATH
export _JAVA_OPTIONS=${_JAVA_OPTIONS:--Xmx2g}   # memory per apktool process
if [ -f "$DH_TOOLS/miniconda3/bin/activate" ]; then
  source "$DH_TOOLS/miniconda3/bin/activate" droidhunter
else
  echo "Environment not installed yet: run bash $DH_CODE/scripts/install.sh" >&2
fi
