import os
import argparse
from typing import List
import pandas as pd
from itertools import chain
from pymedtermino.snomedct import *
import numpy as np
from utils.helpers import save_to_pkl
from tqdm import tqdm


MAPPING_METHODS = ['string', 'lexical']

def __gen_similar_terms(term: str) -> List[str]:
    """Returns a list of similar terms (including the query term itself) using SNOMED-CT
    
    Keyword arguments:
    term -- (str) query term
    Return: a list of similar terms
    """
    
    sim_concepts = SNOMEDCT.search(f'{src_desc} (disorder)')
    concept_terms = [concept.term.lower().replace('(disorder)', '').strip() for concept in sim_concepts]
    concept_isa_terms = list(chain.from_iterable([concept.terms for concept in sim_concepts]))
    concept_isa_terms = [term.lower().replace('(disorder)', '').strip() for term in concept_isa_terms]

    return list(set([term, *concept_terms, *concept_isa_terms]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to the source file')
    parser.add_argument('--tgt', type=str, required=True, help='Path to the target file')
    parser.add_argument('--out', type=str, required=True, help='Path to the output file')
    parser.add_argument('--method', type=str, help='Method to use for generating maps', default=MAPPING_METHODS[0])
    args = parser.parse_args()

    if not os.path.exists(f'./data/{args.src}') or not os.path.exists(f'./data/{args.tgt}'):
        raise FileNotFoundError(f'File not found: {args.src} or {args.tgt}')
    
    if args.method not in MAPPING_METHODS:
        raise ValueError(f'Method should be one of {MAPPING_METHODS}')
    
    src = pd.read_csv(f'./data/{args.src}', dtype=str)
    tgt = pd.read_csv(f'./data/{args.tgt}', dtype=str)

    maps = {}

    for i in tqdm(range(len(src))):
        src_code = src.iloc[i, 0]
        src_desc = src.iloc[i, 1].lower().replace(',', '')
        
        tgt_code_desc = tgt.iloc[:, 1].tolist()
        tgt_code_desc = [x.lower().replace(',', '') for x in tgt_code_desc]

        if args.method == MAPPING_METHODS[0]:
            ## string comparison method
            comparison = [src_desc == tgt_desc for tgt_desc in tgt_code_desc]

            match_idx = [i for i, x in enumerate(comparison) if x]

            if len(match_idx) == 0:
                ## no match found
                continue

            tgt_code = tgt.iloc[match_idx[0], 0]
            tgt_desc = tgt.iloc[match_idx[0], 1].lower().replace(',', '')
        
        elif args.method == MAPPING_METHODS[1]:
            ## lexical comparison method

            ## get the similar terms for the source code descriptions using snomedct
            src_sim_terms = __gen_similar_terms(src_desc)
            tgt_sim_terms = [__gen_similar_terms(tgt_desc) for tgt_desc in tgt_code_desc]

            ## count how many similar terms are common to current source code description and target code descriptions
            comparison = [len(set(src_sim_terms) & set(tgt_sim_terms[i])) for i in range(len(tgt_sim_terms))]

            if sum(comparison) == 0:
                ## no match found
                continue
            
            ## assign target code with maximum num of common similar terms
            tgt_idx = np.argmax(comparison)
            tgt_code = tgt.iloc[tgt_idx, 0]
            tgt_desc = tgt.iloc[tgt_idx, 1].lower().replace(',', '')
        

        # push it to the maps
        maps[src_code] = {
            'tgt': tgt_code,
            'desc': {
                'src': src_desc,
                'tgt': tgt_desc
            }
        }

        ## save the results into a file
        save_to_pkl(maps, f'./results/baseline/{args.out}.pkl')
            