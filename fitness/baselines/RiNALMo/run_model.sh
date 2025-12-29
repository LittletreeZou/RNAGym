conda activate Rinalmo

# Get the current index from the array
# DMS_index=${SLURM_ARRAY_TASK_ID}

DMS_index=0

python compute_fitness.py \
  --reference_sheet /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness/reference_sheet_final.csv \
  --task_id $DMS_index