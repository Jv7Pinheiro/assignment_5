#!/bin/bash
#SBATCH --job-name="PART2"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128GB
#SBATCH --time=00-02:00:00
#SBATCH --output="./Classifier-Part2a-%j.o"
#SBATCH --mail-user=deolj19@wfu.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=csc373
#SBATCH --partition=small

# Load a software module
module load /deac/csc/classes/csc373/software/modulefiles/csc373/python/3.11.8

# Go into your work path
cd /deac/csc/classes/csc373/${USER}/assignment_5/scripts

# Run the program
# python assignment5_part1_estimation.py
# python assignment5_part2_classification.py
python assignment5_part2_classification.py
