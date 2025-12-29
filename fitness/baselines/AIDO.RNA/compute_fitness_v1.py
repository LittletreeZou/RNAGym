"""Adapted from RiNALMo folder
Note that AIDO.RNA was pretrained with T instead of U, take care of the base.
Will add special tokens at the begining and the end the sequence
Mask token id: 1
"""

import torch
import pandas as pd
from modelgenerator.tasks import MLM
import os
import argparse
from scipy import stats
import numpy as np

def get_sequences(wt_sequence, df):
    def apply_mutation(sequence, mutation):
        sequence = sequence.replace('U', 'T')
        possible_bases = ['A', 'T', 'C', 'G', 'N', '']
        mutation = mutation.replace(' ', '')
        pos = int(mutation[1:-1]) - 1
        new_base = mutation[-1]
        old_base = mutation[0]
        old_base = 'T' if old_base == 'U' else old_base
        new_base = 'T' if new_base == 'U' else new_base

        assert old_base in possible_bases, mutation
        assert new_base in possible_bases, mutation

        if old_base != 'N':  
            assert old_base == sequence[pos], mutation
        
        if new_base == '':
            
            mutated_sequence = sequence[:pos] + sequence[pos+1:]
        else:
            
            mutated_sequence = sequence[:pos] + new_base + sequence[pos+1:]

        return mutated_sequence

    def apply_mutations(sequence, mutations):
        for mutation in mutations.split(','):
            sequence = apply_mutation(sequence, mutation)
        return sequence

    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None
    if mutation_column:
        df['mutated_sequence'] = df[mutation_column].apply(lambda x: apply_mutations(wt_sequence, x))
    else:
        raise ValueError("No 'mutant' or 'mutation' column found in the DataFrame")
    
    return df

def apply_masked_marginal_scoring(wt_sequence, mutated_sequence, positions, model, device):
    
    mask_token = model.backbone.tokenizer.mask_token
    mask_idx = model.backbone.tokenizer.token_to_id(mask_token)

    total_score = 0
    possible_bases = ['A', 'T', 'C', 'G']

    for pos in positions:
        
        input = model.transform({"sequences": [mutated_sequence]})
        input['input_ids'][0, pos+1] = mask_idx   # +1 for [BOS]
        
        with torch.no_grad():
            token_logits = model(input)   # (1, 1+seq_len+1, 16), TODO: DON'T USE FP16
        
        # Calculate log probabilities
        token_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        
        # Get log probability of the mutated base at the position
        mut_encoded = model.backbone.tokenizer.token_to_id(mutated_sequence[pos])
        log_prob_mut = token_probs[0, pos + 1, mut_encoded].item()
    
        #remove N's from WT sequence 
        wt_sequence = wt_sequence.replace('N', '')
        
        wt_encoded = model.backbone.tokenizer.token_to_id(wt_sequence[pos])
        log_prob_wt = token_probs[0, pos + 1, wt_encoded].item()
        
        # Calculate the difference and add to the total score
        total_score += log_prob_mut - log_prob_wt

    return total_score

def extract_positions(mutation, offset=1):
    positions = []
    for mut in mutation.split(','):
        pos = int(mut.strip()[1:-1]) - offset
        positions.append(pos)
    return positions

def process_single_row(row, model, device, base_dir, results_dir, score_column):
    dataset = row['DMS_ID']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=['mutant', "dms_score", "sequence"])
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Number of samples: {len(df)}")

    wt_seq = row['RAW_CONSTRUCT_SEQ'].upper().replace("U", "T")
    sequences = get_sequences(wt_seq, df)

    logit_scores = []
    for i, seq in sequences.iterrows():
        mutation_column = 'mutant' if 'mutant' in sequences.columns else 'mutation' if 'mutation' in sequences.columns else 'mutations' if 'mutations' in sequences.columns else None
        positions = extract_positions(seq[mutation_column])
        logits = apply_masked_marginal_scoring(wt_seq,seq['mutated_sequence'], positions, model, device)
        logit_scores.append(logits)
        
    sequences[score_column] = logit_scores
    sequences['mutated_sequence'] = sequences['mutated_sequence'].apply(lambda x: x.replace("T", "U"))
    output_file = os.path.join(results_dir, f"{dataset}.csv")
    sequences.to_csv(output_file, index=False)


def main(args):
    
    DEVICE = "cuda:0"
    
    model = MLM.from_config({"model.backbone": args.model_name})
    model = model.to(device=DEVICE)
    model.eval()

    base_dir = '/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/processed_DMS_files'
    results_dir = '/lustre/scratch/shared-folders/bio_project/shuxian/gbft/mg/rna_202507/RNAGym/fitness_prediction/model_predictions/aidorna-1.6b_results'
    score_column = 'logit_scores'
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs = wt_seqs.rename(columns={"ï»¿DMS_ID": "DMS_ID"})
   
    # wt_seqs.dropna(inplace=True)
    wt_seqs['YEAR'] = wt_seqs['YEAR'].astype(int)
    wt_seqs['AUTHOR'] = wt_seqs['AUTHOR'].astype(str)
    wt_seqs['RNA_TYPE'] = wt_seqs['RNA_TYPE'].astype(str)

    # Select the row corresponding to the task ID
    row = wt_seqs.iloc[args.task_id]
    print(row.DMS_ID)

    process_single_row(row, model, DEVICE, base_dir,results_dir, score_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--model_name', type=str, default="aido_rna_1b600m")
    args = parser.parse_args()
    main(args)