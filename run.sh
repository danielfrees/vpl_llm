#!/bin/bash

sbatch submit_job.sh both 1 base
sbatch submit_job.sh both 1 categorical
sbatch submit_job.sh both 1 mean_and_variance

sbatch submit_job.sh helpful 2 base
sbatch submit_job.sh harmless 2 base

sbatch submit_job.sh helpful 2 categorical
sbatch submit_job.sh harmless 2 categorical

sbatch submit_job.sh helpful 2 mean_and_variance
sbatch submit_job.sh harmless 2 mean_and_variance