echo ">>> RNAGym zeroshot fitness prediction >>>"

MODEL=aido_rna_650m_cds  #aido_rna_1b600m_cds
METHOD=masked-marginals

for dms_idx in {0..69}
do
    log_dir=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/logs/${MODEL}-${METHOD}/$dms_idx

    sbatch --output=${log_dir}.out --error=${log_dir}.err \
        run_v2.sh $MODEL $METHOD $dms_idx

    # sleep 1
done
