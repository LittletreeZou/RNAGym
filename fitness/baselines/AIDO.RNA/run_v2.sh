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

MODEL=aido_rna_650m    #$1   #aido_rna_1b600m_cds
METHOD=wt-marginals      #masked-marginals

echo "${MODEL} ${METHOD} ${DMS_IDX}"

ref=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness/reference_sheet_final.csv
dms=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/processed_DMS_files
out=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/model_predictions/${MODEL}-${METHOD}

mkdir -p ${out}

if [ $METHOD == "masked-marginals" ]; then
    DMS_IDX=$3
    srun python compute_fitness_v2.py  \
        --model_name $MODEL \
        --reference_sequences $ref \
        --dms_directory $dms \
        --output_directory $out\
        --scoring-strategy $METHOD \
        --dms_idx $DMS_IDX
else
    CUDA_VISIBLE_DEVICES=2 python compute_fitness_v2.py  \
        --model_name $MODEL \
        --reference_sequences $ref \
        --dms_directory $dms \
        --output_directory $out\
        --scoring-strategy $METHOD
fi