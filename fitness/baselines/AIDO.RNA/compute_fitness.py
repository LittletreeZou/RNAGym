import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from modelgenerator.tasks import MLM


def clean_sequence(seq, assay_name): 
    # replaces invalid bases with N
    cleaned = ''.join([c if c in {'A', 'T', 'C', 'G'} else 'N' for c in seq])
    if cleaned != seq:
        print(f"[WARNING] Cleaned WT sequence for assay '{assay_name}' due to invalid characters.")
        print(f"Original: {seq}")
        print(f"Cleaned:  {cleaned}")
    return cleaned


def label_row_wt(row, full_sequence, full_token_probs, offset_idx, assay_name, model, max_len=1024): 
    """scoring with wt-marginals strategy"""

    score = 0
    seq_len = len(full_sequence)

    # Parse all mutations in the row
    try:
        mutations = [m.strip() for m in row.split(",")]
        mutation_info = [(m[0], int(m[1:-1]) - offset_idx, m[-1]) for m in mutations]
    except Exception as parse_error:
        print(f"[EXCEPTION] Failed to parse mutation row '{row}' in assay '{assay_name}'")
        print(parse_error)
        return np.nan

    try:
        # Sanity check: WT matches sequence
        for wt, idx, mt in mutation_info:
            if idx < 0 or idx >= seq_len:
                raise IndexError(
                    f"\n[ERROR] Mutation index {idx} out of bounds for sequence length {seq_len} in assay: {assay_name}"
                )
            if full_sequence[idx] != wt:
                raise ValueError(
                    f"\n[ERROR] Wildtype base mismatch in assay: {assay_name}\n"
                    f"  Mutation: {wt}{idx + offset_idx}{mt}\n"
                    f"  Expected WT: {wt} at position {idx}\n"
                    f"  Actual base in sequence: {full_sequence[idx]}\n"
                    f"  Full sequence: {full_sequence}\n"
                    f"  Problematic mutation row: {row}\n"
                )

        # If the full sequence fits in the model
        if seq_len + 2 <= max_len and full_token_probs is not None:
            for wt, idx, mt in mutation_info:
                wt_encoded = model.backbone.tokenizer.token_to_id(wt)
                mt_encoded = model.backbone.tokenizer.token_to_id(mt)
                score += (full_token_probs[0, 1 + idx, mt_encoded] - full_token_probs[0, 1 + idx, wt_encoded]).item()
        else:
            # Use windowing for sequences longer than maximum input length centered on mutations
            idxs = [idx for _, idx, _ in mutation_info]
            min_idx, max_idx = min(idxs), max(idxs)
            window_size = max_len
            window_half = (window_size - 2) // 2  # excluding BOS and EOS

            center = (min_idx + max_idx) // 2
            start_pos = max(0, center - window_half)
            end_pos = min(seq_len, start_pos + (window_size - 2))
            if end_pos == seq_len:
                start_pos = max(0, seq_len - (window_size - 2))
                end_pos = seq_len

            if not all(start_pos <= idx < end_pos for idx in idxs):
                raise ValueError(
                    f"[ERROR] Window does not include all mutations.\n"
                    f"  Mutations: {mutation_info}\n"
                    f"  Window range: {start_pos}-{end_pos - 1}\n"
                    f"  Assay: {assay_name}"
                )

            window_seq = full_sequence[start_pos:end_pos]
            input = model.transform({"sequences": [window_seq]})

            with torch.no_grad():
                token_logits = model(input)
                token_probs = torch.log_softmax(token_logits, dim=-1)

            for wt, idx, mt in mutation_info:
                window_idx = idx - start_pos
                if not (0 <= window_idx < end_pos - start_pos):
                    raise IndexError(
                        f"[IndexError] window_idx={window_idx} is out of bounds for mutation {wt}{idx+offset_idx}{mt} "
                        f"in window [{start_pos}, {end_pos})"
                    )
                wt_encoded = model.backbone.tokenizer.token_to_id(wt)
                mt_encoded = model.backbone.tokenizer.token_to_id(mt)
                score += (token_probs[0, 1 + window_idx, mt_encoded] - token_probs[0, 1 + window_idx, wt_encoded]).item()

    except Exception as e:
        print(f"[EXCEPTION] Assay '{assay_name}', Mutation row: {row}")
        print(e)
        return np.nan

    return score


def compute_scores_wt(model, reference, args):
   
    for _, row in reference.iterrows():
        name = row['DMS_ID']

        # Avoid repeated computation
        if os.path.exists(f"{args.output_directory}/{name}.csv"):
            continue

        wt_rna = row['RAW_CONSTRUCT_SEQ'].upper().replace('U', 'T')
        wt_rna = clean_sequence(wt_rna, name)
        csv_path = row['PATH']
        seq_len = len(wt_rna)

        try:
            if seq_len + 2 <= 1024:
                # Full sequence fits
                input = model.transform({"sequences": [wt_rna]})
                with torch.no_grad():
                    token_logits = model(input)
                    token_probs = torch.log_softmax(token_logits, dim=-1)
            else:
                # Will trigger windowing
                token_probs = None  

        except Exception as model_error:
            print(f"\n[ERROR] Model inference failed for assay '{name}'")
            print(f"  Exception type : {type(model_error).__name__}")
            print(f"  Error message  : {model_error}")
            print(f"  Sequence length: {seq_len}")
            continue

        try:
            mut_df = pd.read_csv(csv_path)
            mut_df = mut_df[~mut_df['mutant'].isna()]
            mut_df['mutant'] = (
                mut_df['mutant']
                .astype(str)
                .str.upper()
                .str.strip()
                .str.replace("U", "T")
                .str.replace(" ", "")
            )

            valid_mask = mut_df['mutant'].str.match(r'^([ATCG][0-9]+[ATCG])(,[ATCG][0-9]+[ATCG])*$')
            if not valid_mask.all():
                print(f"[WARNING] Invalid mutation format found in assay '{name}':")
                print(mut_df[~valid_mask])
            mut_df = mut_df[valid_mask]
            mut_list = mut_df['mutant'].to_list()

            scores = []
            failed_mutations = []

            for mut in tqdm(mut_list, desc=f"Scoring mutations ({name})"):
                try:
                    score = label_row_wt(
                        row=mut,
                        full_sequence=wt_rna,
                        full_token_probs=token_probs,
                        offset_idx=1,
                        assay_name=name,
                        model=model,
                        max_len=1024
                    )
                    scores.append(score)
                except Exception as mutation_error:
                    print(f"[ERROR] Failed to score mutation '{mut}' in assay '{name}':")
                    print(mutation_error)
                    failed_mutations.append(mut)
                    scores.append(np.nan)

            if failed_mutations:
                print(f"\n[SUMMARY] Assay '{name}' had {len(failed_mutations)} failed mutations:")
                for fm in failed_mutations:
                    print(f"  ➤ {fm}")
                print("-" * 60)

            mut_df['logit_scores'] = scores
            score_path = f'{args.output_directory}/{name}.csv'
            mut_df.to_csv(score_path, index=False)
            print(f"File saved at {score_path}")

        except Exception as e:
            print(f"\n[ERROR] Failed during mutation processing for assay '{name}'")
            print(f"  Exception type : {type(e).__name__}")
            print(f"  Error message  : {e}")
            print("-" * 80)



def compute_scores_masked_multiple(model, row, args, batch_size=64, max_len=1024): 
    """
    scoring with masked-marginals strategy

    Args:
        row: row in reference sheet
    """
    mask_token = model.backbone.tokenizer.token_to_id(model.backbone.tokenizer.mask_token)

    name = row['DMS_ID']
    wt_rna = row['RAW_CONSTRUCT_SEQ'].upper().replace('U', 'T')
    wt_rna = clean_sequence(wt_rna, name)
    csv_path = row['PATH']

    try:
        mut_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read mutation CSV for {name}: {e}")

    mut_df = mut_df[~mut_df['mutant'].isna()]
    mut_df['mutant'] = (
            mut_df['mutant']
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace("U", "T")
            .str.replace(" ", "")
        )

    mut_list = mut_df['mutant'].tolist()
    mutation_info = []
    for i, mut in enumerate(tqdm(mut_list, desc=f"{name} mutations")):
        # try:
        # mutiple mutations
        mutations = [m.strip() for m in mut.split(",")]
        idxs = []
        for m in mutations:
            wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
            if idx >= len(wt_rna) or wt_rna[idx] != wt:
                raise ValueError(f"[SKIP] WT mismatch at {m} in {name}")
            idxs.append(idx)

        if len(wt_rna) + 2 <= max_len:
            window_seq = wt_rna
            start_pos = 0
        else:
            min_idx, max_idx = min(idxs), max(idxs)
            center = (min_idx + max_idx) // 2
            window_half = (max_len - 2) // 2
            start_pos = max(0, center - window_half)
            end_pos = min(len(wt_rna), start_pos + (max_len - 2))
            if end_pos == len(wt_rna):
                start_pos = max(0, len(wt_rna) - (max_len - 2))
            window_seq = wt_rna[start_pos:end_pos]
            if not all(start_pos <= idx < end_pos for idx in idxs):
                raise ValueError(f"[SKIP] Not all mutation sites are within model window for {name}: {mut}")

        tokenized_input = model.transform({"sequences": [window_seq]})
        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]
        special_tokens_mask = tokenized_input["special_tokens_mask"]
        
        # masked all the mutation positions
        masked_input_ids = input_ids.clone()
        for m in mutations:
            wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
            window_idx = idx - start_pos
            masked_input_ids[0, window_idx + 1] = mask_token
        
        masked_input = {"input_ids": masked_input_ids, 
                        "attention_mask": attention_mask,
                        "special_tokens_mask": special_tokens_mask,
                        }
        mutation_info.append((i, mutations, masked_input, start_pos))

        # except Exception as e:
        #     print(f"[ERROR] Skipping mutation '{mut}' in assay '{name}': {e}")
        #     mutation_info.append((i, None, None, None))

    # compute the score
    valid_entries = [entry for entry in mutation_info if entry[1] is not None]
    scores = [np.nan] * len(mut_list)

    for i in range(0, len(valid_entries), batch_size):
        batch = valid_entries[i: i + batch_size]

        # Collect all masked_input dicts from this batch
        masked_input_batch = [entry[2] for entry in batch]
        batch_input = {
                    k: torch.cat([x[k] for x in masked_input_batch], dim=0)
                    for k in masked_input_batch[0].keys()
                }

        with torch.no_grad():
            logits = model(batch_input)
            log_probs = torch.log_softmax(logits, dim=-1)

        for j, (row_idx, mutations, _, start_pos) in enumerate(batch):
            try:
                score = 0
                for m in mutations:
                    wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
                    window_idx = idx - start_pos
                    wt_idx = model.backbone.tokenizer.token_to_id(wt)
                    mt_idx = model.backbone.tokenizer.token_to_id(mt)
                    score += (log_probs[j, window_idx + 1, mt_idx] - log_probs[j, window_idx + 1, wt_idx]).item()
                scores[row_idx] = score
            except Exception as e:
                print(f"[ERROR] Failed scoring mutation {mut_list[row_idx]} in assay '{name}': {e}")
                scores[row_idx] = np.nan

    mut_df['logit_score'] = scores
    score_path = f'{args.output_directory}/{name}.csv'
    try:
        mut_df.to_csv(score_path, index=False, encoding='utf-8')
        print("File saved at", score_path)
    except Exception as e:
        print(f"[ERROR] Failed to save scored file for {name}: {e}")


def create_parser():
    parser = argparse.ArgumentParser(description='Label an RNA mutation dataset with zero-shot predictions from AIDO.RNA')
    parser.add_argument("--reference_sequences", type=str, help="CSV file with reference sequences")
    parser.add_argument("--dms_idx", type=int, help="index for dms in reference sheet, from 0-69")
    parser.add_argument("--dms_directory", type=str, help="Directory of mutational datasets")
    parser.add_argument("--output_directory", type=str, help="Directory to save scored fitness files")
    parser.add_argument("--scoring-strategy", type=str, default="masked-marginals", choices=["wt-marginals", "masked-marginals"], help="Scoring strategy")
    parser.add_argument('--model_name', type=str, default="aido_rna_1b600m")
    return parser


def main(args):

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = MLM.from_config({"model.backbone": args.model_name})
    model = model.to(device=DEVICE)
    model.eval()

    reference = pd.read_csv(args.reference_sequences)
    reference['PATH'] = reference['DMS_ID'].apply(lambda x: f"{args.dms_directory}/{x}.csv")

    if args.scoring_strategy == "wt-marginals":
        compute_scores_wt(model, reference, args)
    elif args.scoring_strategy == "masked-marginals":
        row = reference.iloc[args.dms_idx]
        print(row['DMS_ID'])
        compute_scores_masked_multiple(model, row, args)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
