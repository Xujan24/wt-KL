import os
import sys
from pathlib import Path
import argparse
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.helpers import save_to_pkl, load_from_pkl, get_embeddings, get_similar_terms, get_similar_terms_using_hierarchy, gen_paraphrases
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to generate embeddings for ICD codes.")
    parser.add_argument("--input", type=str, help="Path to .csv file")
    parser.add_argument("--use_similar_terms", help="Whether to use similar terms or not", action="store_true")
    parser.add_argument("--similar_terms_only", help="Whether to use only similar terms or not", action="store_true")
    parser.add_argument("--local_similar_terms", action="store_true", help="Whether to use local similar terms or create one using SNOMEDCT")
    parser.add_argument("--model_ver", help="Version of the pretrained model", type=str, default="base")
    parser.add_argument("--d", type=int, help="if --use_similar_terms set to True, then number of similar terms for each code")
    parser.add_argument("--save_similar_terms", help="if --use_similar_terms is set to True then whether to save the similar terms or not", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = 'all-MiniLM-L6-v2' if args.model_ver == 'base' else f'./examples/models/all-MiniLM-L6-v2/final/{args.model_ver}'

    if args.input is None:
        raise ValueError("--input and --output arguments are required.")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} doesn't exist.")
    
    df = pd.read_csv(args.input, dtype=str)
    ## make sure that the dataframe has column name 'code' and 'code_desc' for ICD code and code descriptor respectively
    columns = df.columns.to_list()
    if 'code' not in columns or 'code_desc' not in columns:
        raise ValueError("Dataframe should have 'code' and 'code_desc' columns.")
    
    del columns

    ## get the base filename without extension from naming the output files
    basename = Path(args.input).stem

    terms = {}

    tqdm.write(f"Extracting (or Generating) terms (or similar terms) for {basename}...")
    if args.use_similar_terms:
        if args.local_similar_terms:
            terms = load_from_pkl(f'./data/similar_terms/{basename}-terms-50.pkl')
        else:
            ## load pretrained model and tokenizer for generating paraphrases when required
            tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
            model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
            
            for i in tqdm(range(len(df))):
                code = df.iloc[i, 0]
                code_desc = df.iloc[i, 1].lower()
                
                terms[code] = get_similar_terms(x = code_desc, n = args.d, model=model, tokenizer=tokenizer, device=device)

            if args.save_similar_terms:
                save_dir = './data/similar_terms'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_to_pkl(terms, f'{save_dir}/{basename}-terms-{args.d}.pkl')

            del tokenizer, model
            torch.cuda.empty_cache()

    else:
        for i in tqdm(range(len(df))):
            code = df.iloc[i, 0]
            code_desc = df.iloc[i, 1].lower()

            terms[code] = code_desc

    if args.similar_terms_only:
        sys.exit(0)
    
    ## iniitalize the sentence transformer model
    model_st = SentenceTransformer(model_id)
    model_st.eval()

    terms_d = {}
    tqdm.write(f"Generating embeddings for {basename}...")
    progress = tqdm(total=len(terms), position=0, leave=True)
    for k, v in terms.items():
        if not args.use_similar_terms:
            emb = model_st.encode(v)
            terms_d[k] = emb
            progress.update(1)
            continue

        inputs = v[:args.d]
        out, mu, cov = get_embeddings(model_st, inputs, device)
        terms_d[k] = {
            'inputs': out,
            'mu': mu, 
            'cov': cov
        }
        progress.update(1)

    
    del model_st
    torch.cuda.empty_cache()

    ## prepare the file to save the embeddings
    out_dir = './data/inputs' if args.model_ver == 'base' else f'./data/inputs/{args.model_ver}'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = f'{out_dir}/{basename}-{args.d}.pkl' if args.use_similar_terms else f'{out_dir}/{basename}-emb.pkl'
    save_to_pkl(terms_d, out_file)