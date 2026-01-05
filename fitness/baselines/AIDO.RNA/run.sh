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

MODEL=aido_rna_1b600m_cds_3096      #$1
METHOD=wt-marginals      #masked-marginals
MAX_LEN=3096

echo "${MODEL} ${METHOD} ${MAX_LEN} ${DMS_IDX}"

WORK_DIR=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym
ref=${WORK_DIR}/fitness/reference_sheet_final.csv
dms=${WORK_DIR}/fitness_prediction/assays
out=${WORK_DIR}/fitness_prediction/model_predictions_20251117/${MODEL}-${METHOD}

mkdir -p ${out}

if [ $METHOD == "masked-marginals" ]; then
    DMS_IDX=$3
    srun python compute_fitness.py  \
        --model_name $MODEL \
        --reference_sequences $ref \
        --dms_directory $dms \
        --output_directory $out\
        --scoring-strategy $METHOD \
        --dms_idx $DMS_IDX \
        --max_seq_len $MAX_LEN
else
    CUDA_VISIBLE_DEVICES=1 python compute_fitness.py  \
        --model_name $MODEL \
        --reference_sequences $ref \
        --dms_directory $dms \
        --output_directory $out\
        --scoring-strategy $METHOD \
        --max_seq_len $MAX_LEN
fi
