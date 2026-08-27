#!/bin/bash
#SBATCH --job-name=stresstest-resume
#SBATCH --output=stresstest_large_resume.out
#SBATCH --time=24:00:00

#SBATCH --mail-type=BEGIN,END,FAIL    # Specify which CPU/job events trigger an email
#SBATCH --mail-user=ahmedn@crick.ac.uk  # Your actual email address

# Resume of run `condescending_jennings` (session 06f0a510), which lost combineStacks to
# the 137/140 loop. The 16 completed runModel tasks are cached and will be skipped;
# combineStacks failed, so it re-runs and picks up the fixed combine_stacks.py.
#
# --time is 24h, not the 3h the original used: that 3h limit is what cancelled the head
# job at 18:44 while its combineStacks child kept running to 21:05, so the pipeline lost
# track of a job that was still alive.

ml Nextflow/24.04.1 && ml Mamba && nextflow run . --img_dir /nemo/stp/ddt/working/ahmedn/aiod-312/all_img_paths.csv --model empanada --model_type mitonet_v1 --task mito -profile crick --profile_memory true -c ../overrides.config -resume 06f0a510-ec14-40c2-aa0b-f2bc928a1c1c
