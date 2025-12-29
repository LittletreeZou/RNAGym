from performance_fitness import calculate_metrics
import pandas as pd
import os


score_cols_dict = {
        'evo1': 'evo_1_131k_base_score',
        'evo1.5': 'evo_1.5_8k_base_score',
        'evo2': 'evo2_7b_score',
        'GenSLM': 'logit_scores',
        'NT': 'kmer_pseudo_LL',
        'RNA-FM': 'RNA_FM_scores',
        'rinalmo': 'logit_scores',
        'RNAErnie': 'Mutation_Scores',
        'aido_rna_650m-wt-marginals': 'logit_scores',
        'aido_rna_650m_cds-wt-marginals': 'logit_scores',
        'aido_rna_1b600m-wt-marginals': 'RNA_FM_scores',
        'aido_rna_1b600m_cds-wt-marginals': 'logit_scores'
    }


if __name__ == "__main__":

    reference_sheet = "/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness/reference_sheet_final.csv"
    ref_df = pd.read_csv(reference_sheet)

    model_list = ['evo1','evo1.5','evo2','GenSLM', 'NT','rinalmo','RNAErnie','RNA-FM']
    # model_list = ["aido_rna_1b600m-wt-marginals"]

    for model in model_list:
        print(model)
        score_col = score_cols_dict[model]
    
        model_prediction_dir=f"/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/model_predictions/{model}"

        scores = []
        for _, row in ref_df.iterrows():
            dms_id = row['DMS_ID']
            input_file = os.path.join(model_prediction_dir, f"{dms_id}.csv")
            if os.path.exists(input_file):
                df = pd.read_csv(input_file)

                gt_score = df['DMS_score'].values
                pred_score = df[score_col].values

                score = calculate_metrics(gt_score, pred_score)
                sp = score["Spearman"]
                auc = score["AUC"]
                mcc = score["MCC"]

                scores.append((dms_id, sp, auc, mcc))
            else:
                print(f"Predictions for {dms_id} not found")

        score_df = pd.DataFrame(scores, columns=["dms_id", "spearman", "auc", "mcc"])
        save_dir = "/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/metric"
        save_path = f"{save_dir}/{model}.csv"
        
        score_df.to_csv(save_path)

        import pdb
        pdb.set_trace()
