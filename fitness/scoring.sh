# python merge_scoring_files.py \
#     --processed_folder /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/processed_DMS_files \
#     --model_predictions_folder /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/model_predictions \
#     --output_folder /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction


python performance_fitness.py \
    --reference_file /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness/reference_sheet_final.csv \
    --combined_dir /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/model_predictions/aidorna-1.6b_results \
    --performance_dir /lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/performance
