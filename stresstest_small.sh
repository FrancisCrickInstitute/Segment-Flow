#!/bin/bash
#SBATCH --job-name=stresstest_small
#SBATCH --output=stresstest_small.out
#SBATCH --time=00:30:00

#SBATCH --mail-type=BEGIN,END,FAIL    # Specify which CPU/job events trigger an email
#SBATCH --mail-user=ahmedn@crick.ac.uk  # Your actual email address

ml Nextflow/24.04.1 && ml Mamba && nextflow run . --img_dir /nemo/stp/ddt/working/ahmedn/aiod-312/small_image.csv --model empanada --model_type mitonet_v1 --task mito --num_substacks "1,1,2" -profile crick --profile_memory true -c ../overrides.config
