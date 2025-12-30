WORK_DIR=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym

# STEP 1
python merge_scoring_files.py \
    --processed_folder ${WORK_DIR}/fitness_prediction/assays \
    --model_predictions_folder ${WORK_DIR}/fitness_prediction/model_predictions_20251117 \
    --output_folder ${WORK_DIR}/fitness_prediction/merged_files

# STEP 2
python performance_fitness.py \
    --reference_file /${WORK_DIR}/fitness/reference_sheet_final.csv \
    --combined_dir ${WORK_DIR}/fitness_prediction/merged_files \
    --performance_dir ${WORK_DIR}/fitness_prediction/performance
