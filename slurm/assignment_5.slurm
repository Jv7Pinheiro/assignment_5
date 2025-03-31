#!/bin/bash
#SBATCH --job-name="PART1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=00-02:00:00
#SBATCH --output="./DOGO_FR-%j.o"
#SBATCH --error="./DOGO_ERROR-%j.e"
#SBATCH --mail-user=gonza224@wfu.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

# Load a software module
module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8

# Go into your work path
cd /deac/csc/classes/csc373/${USER}/assignment_5/scripts

# Run the program
# python ... ... ...
# python assignment_3.py
python assignment5_part1_estimation.py
