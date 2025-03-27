#!/bin/bash
#SBATCH --job-name="PART1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=50GB
#SBATCH --time=00-01:00:00
#SBATCH --output="DOGO_FR-%j.o"
#SBATCH --error="DOGO_ERROR-%j.e"
#SBATCH --mail-user=deolj19@wfu.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=ballardGrp
#SBATCH --partition=small

# Load a software module
module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8

# Go into your work path
cd /deac/csc/classes/csc373/${USER}/assignment_5/scripts

# Run the program
# python ... ... ...
# python assignment_3.py
python assignment_5_part1_estimation.py
