#!/bin/bash
#SBATCH --job-name=stresstest
#SBATCH --output=stresstest_large.out
#SBATCH --time=03:00:00

#SBATCH --mail-type=BEGIN,END,FAIL    # Specify which CPU/job events trigger an email
#SBATCH --mail-user=ahmedn@crick.ac.uk  # Your actual email address

# Full EMPIAR-scale run, combineStacks wrapped in memray so a real flamegraph + .bin
# are published next to the output mask. See stresstest_small.sh for the small-image
# dry run this follows up on.
ml Nextflow/24.04.1 && ml Mamba && nextflow run . --img_dir /nemo/stp/ddt/working/ahmedn/aiod-312/all_img_paths.csv --model empanada --model_type mitonet_v1 --task mito -profile crick --profile_memory true -c ../overrides.config
