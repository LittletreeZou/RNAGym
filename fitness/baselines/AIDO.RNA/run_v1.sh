#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=bio
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH -p gpumid
#SBATCH --job-name RNAGym
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e 

# conda activate mgen
export HF_HOME=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/hf_home

DMS_index=$1

echo "Running task ${DMS_index}"

# srun 
python compute_fitness.py \
  --reference_sheet /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness/reference_sheet_final.csv \
  --model_name aido_rna_1b600m \
  --task_id $DMS_index
