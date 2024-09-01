"""
Python script to generate mappings using cosine similarity and KL divergence based methods.
"""
import os
from pathlib import Path
from typing import Dict, Tuple, Union
import pandas as pd
from utils.helpers import save_to_pkl, load_from_pkl
from utils.sim import kl_div
from sentence_transformers import util
import numpy as np
import argparse
from itertools import product
from tqdm import tqdm

INPUT_PATH = './data/inputs'
K = [1, 2, 3, 5, 7, 10 ]


def code_to_idx(df: pd.DataFrame, code: str) -> Union[int, None]:
    """Converts a code to an index assuming given dataframe has a column with name 'code'

    Keyword arguments:
        df (pd.DataFrame) -- a dataframe
        code (str) -- a code
    Return:
        an index
    """
    code_list = df['code'].to_list()
    return int(code_list.index(code)) if code in df['code'].to_list() else None


def compute_kl_scores(src_inputs_d: Dict, tgt_inputs_d: Dict) -> Dict:
    """Computes KL divergence scores between source and target embeddings
    
        Keyword Arguments:
            src_inputs_d (Dict) -- A dictonary for source embeddings
            tgt_inputs_d (Dict) -- A dictonary for target embeddings 
        Return: 
            a dictonary of KL scores
    """

    progress = tqdm(total=len(src_inputs_d), position=0, leave=True)
    
    kl_scores = {}
    for k, v in src_inputs_d.items():
        result = []
        for _, _v in tgt_inputs_d.items():
            result.append( kl_div(_v, v))
        kl_scores[k] = result
        del result
        progress.update(1)
    
    progress.close()
    del progress
    return kl_scores


def compute_cs_scores(src_emb: np.ndarray, tgt_emb: np.ndarray) -> np.ndarray:
    """Computes cosine similarity scores between source and target embeddings
    
    Keyword arguments:
        src_emb (np.ndarray) -- source embeddings
        tgt_emb (np.ndarray) -- target embeddings
    Return: 
        a numpy.ndarray of cosine similarity scores
    """
    return util.cos_sim(src_emb, tgt_emb).numpy()


def gen_map(scores: np.ndarray, reverse: bool = False) -> np.ndarray:
    '''
    Given the scores, returns a mapping of source to target indices.
    Keyword arguments:
        scores (np.ndarray) -- a numpy array of scores
        reverse (bool) -- whether to sort in reverse order. If True, returns the result in descending order. Default is False 
    '''
    res = np.argsort(scores, axis=1)
    if reverse:
        res = np.fliplr(res)
    
    return res


def get_maps(src_df: pd.DataFrame, tgt_df: pd.DataFrame, k: int, maps: np.ndarray) -> Dict:
    res = {}

    for i in range(len(src_df)):
        src_code = src_df['code'].tolist()[i]
        src_code_desc = src_df['code_desc'].tolist()[i]
        
        m_codes = [tgt_df['code'].tolist()[x] for x in maps[i][:k]]
        m_code_desc = [tgt_df['code_desc'].tolist()[x] for x in maps[i][:k]]

        res[src_code] = {
            'code_desc': src_code_desc,
            'm_codes': m_codes,
            'm_code_desc': m_code_desc
        }
    
    return res


def compute_statistics(
        src_df: pd.DataFrame, 
        tgt_df: pd.DataFrame, 
        maps: np.ndarray, 
        gt_dict: Dict, k: int, 
        gt_scores: np.ndarray
    ) -> Tuple[int, int]:
    """Computes accuracy for a given set of scores

    Keyword arguments:
        src_df (pd.DataFrame) -- source dataframe
        tgt_df (pd.DataFrame) -- target dataframe
        maps (np.ndarray) -- a numpy array of sorted indices for source to target mappings
        gt_dict (Dict) -- a dictionary of ground truth labels
        k (int) -- k value for top k accuracy
        reverse (bool) -- whether to sort in reverse order
    Return:
        a tuple of correct predictions and not in ground truth
    """
    correct = 0
    not_in_gt = 0
    return_score = 0

    for i in range(len(src_df)):
        src = src_df.iloc[i, 0]
        gt = gt_dict.get(src)

        if gt is None:
            not_in_gt += 1
            continue

        m = [tgt_df.iloc[x, 0] for x in maps[i][:k]]

        if len(set(m) & set(gt)) > 0:
            correct += 1

        # calculate scores
        gt_idx = [code_to_idx(tgt_df, x) for x in gt if code_to_idx(tgt_df, x) is not None]
        pred_idx = [code_to_idx(tgt_df, x) for x in m if code_to_idx(tgt_df, x) is not None]

        if len(gt_idx) == 0 or len(pred_idx) == 0:
            continue

        s = [gt_scores[x][y] for x, y in product(gt_idx, pred_idx)]
        s = max(s)
        return_score += s
    
    return correct, not_in_gt, return_score


if __name__ == "__main__":
    ## required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="path to source system e.g. ./data/icd9cm.csv")
    parser.add_argument("--tgt", type=str, help="path to target system e.g. ./data/icd10cm.csv")
    parser.add_argument("--d", type=int, help="number of similar terms used to calculate the parameters of the distributions")
    parser.add_argument("--model_ver", type=str, default='base', help="model version to use for sentence embeddings")
    parser.add_argument("--compute_statistics", action='store_true', help="if set, will compute statistics for the given source and target systems")
    args = parser.parse_args()

    if not args.src or not args.tgt or not args.d:
        raise ValueError('One or more arguments are missing. Valid script python main.py [--src] [--tgt] [--d]')


    ## load source and target dataframes, and corresponding ground truth dictionaries
    if not os.path.exists(args.src) or not os.path.exists(args.tgt):
        raise FileNotFoundError(f'{args.src}.csv or {args.tgt}.csv file doesn\'t exist.')
    
    src_basename = Path(args.src).stem
    tgt_basename = Path(args.tgt).stem
    gt_path = os.path.join('./data/gt', f'{src_basename}_{tgt_basename}.pkl')

    if args.compute_statistics and not os.path.exists(gt_path):
        raise FileNotFoundError(f'{gt_path} file doesn\'t exist.')

    source_df = pd.read_csv(args.src, dtype=str)
    target_df = pd.read_csv(args.tgt, dtype=str)

    if args.compute_statistics:
        gt_dict = load_from_pkl(gt_path)
    
    ## load source and target code description embeddings, used to calculate the cosine similarity
    input_dir = './data/inputs' if args.model_ver == 'base' else f'./data/inputs/{args.model_ver}'
    src_emb_path = os.path.join(input_dir, f'{src_basename}-emb.pkl')
    tgt_emb_path = os.path.join(input_dir, f'{tgt_basename}-emb.pkl')

    if not os.path.exists(src_emb_path) or not os.path.exists(tgt_emb_path):
        raise FileNotFoundError(f'{src_emb_path} and/or {tgt_emb_path} file doesn\'t exist.')
    
    src_emb = load_from_pkl(src_emb_path)
    tgt_emb = load_from_pkl(tgt_emb_path)
    
    ## load source and target probabilistic embeddings, used to calculate KL divergence
    src_inputs_d_path = os.path.join(input_dir, f'{src_basename}-{args.d}.pkl')
    tgt_inputs_d_path = os.path.join(input_dir, f'{tgt_basename}-{args.d}.pkl')

    if not os.path.exists(src_inputs_d_path) or not os.path.exists(tgt_inputs_d_path):
        raise FileNotFoundError(f'{src_inputs_d_path} and/or {tgt_inputs_d_path} file doesn\'t exist.')

    src_inputs_d = load_from_pkl(src_inputs_d_path)
    tgt_inputs_d = load_from_pkl(tgt_inputs_d_path)

    ## load the scores for the target systems
    gt_scores_path = os.path.join('./data/scores', f'{tgt_basename}.pkl')
    if not os.path.exists(gt_scores_path):
        raise ValueError(f'{gt_scores_path} file doesn\'t exist for {tgt_basename}.')
    
    gt_scores = load_from_pkl(gt_scores_path)

    ## prepare output file to log results
    results_dir = os.path.join('./results', f'n-{args.d}')

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    out_dir = os.path.join(results_dir, 'default') if args.model_ver == 'base' else os.path.join(results_dir, args.model_ver)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    outfile_path = os.path.join(out_dir, f'{src_basename}_{tgt_basename}')

    if not os.path.exists(outfile_path):
        os.mkdir(outfile_path)

    if args.compute_statistics:
        out_stats_file = open(os.path.join(outfile_path, 'log.txt'), 'w')
        out_stats_file.write(f'[{args.src} - {args.tgt}]\n\n')


    ## Cosine Similarity based mapping
    tqdm.write('Generating maps using Cosine Similarity...')
    src_emb_mat = np.asarray([v for _, v in src_emb.items()]) if isinstance(src_emb, dict) else src_emb
    tgt_emb_mat = np.asarray([v for _, v in tgt_emb.items()]) if isinstance(tgt_emb, dict) else tgt_emb

    cs_score = compute_cs_scores(src_emb_mat, tgt_emb_mat)
    cs_map = gen_map(cs_score, reverse=True)

    cs_maps = get_maps(source_df, target_df, 10, cs_map)
    save_to_pkl(cs_maps, os.path.join(outfile_path, 'cs_maps.pkl'))

    if args.compute_statistics:
        out_stats_file.write('Method: Cosine similarity between code description embeddings \n')
        for k in K:
            correct, not_found, score = compute_statistics(
                            src_df=source_df,
                            tgt_df=target_df,
                            maps=cs_map,
                            gt_dict=gt_dict,
                            k = k,
                            gt_scores=gt_scores
                        )

            accuracy = round(correct / (len(source_df) - not_found), 4)
            hs_score = round(score / (len(source_df) - not_found), 4)
            out_stats_file.write(f'Top-{k} accuracy: \t {accuracy}, [{correct} | {hs_score} | {not_found} | {len(source_df)}] \n')


    ## KL divergence based mapping
    tqdm.write('Generating maps using KL divergence...')
    kl_scores = compute_kl_scores(src_inputs_d, tgt_inputs_d)
    kl_score_mat = np.asarray([v for _, v in kl_scores.items()])
    kl_map = gen_map(kl_score_mat, reverse=False)

    kl_maps = get_maps(source_df, target_df, 10, kl_map)
    save_to_pkl(kl_maps, os.path.join(outfile_path, 'kl_maps.pkl'))

    if args.compute_statistics:
        out_stats_file.write('\n\nMethod: KL Divergence (using {d} similar (SNOMED-CT) terms) \n')
        for k in K:
            correct, not_found, score = compute_statistics(
                            src_df=source_df,
                            tgt_df=target_df,
                            maps=kl_map,
                            gt_dict=gt_dict,
                            k = k,
                            gt_scores=gt_scores
                        )

            accuracy = round(correct / (len(source_df) - not_found), 4)
            hs_score = round(score / (len(source_df) - not_found), 4)
            out_stats_file.write(f'Top-{k} accuracy: \t {accuracy}, [{correct} | {hs_score} | {not_found} | {len(source_df)}] \n')

    ## weighted KL divergence based mapping
    tqdm.write('Generating maps using Weighted KL divergence...')
    cs_dist = 1 - cs_score
    weighted_kl_scores = np.multiply(cs_dist, kl_score_mat)
    weighted_kl_map = gen_map(weighted_kl_scores, reverse=False)

    wt_kl_maps = get_maps(source_df, target_df, 10, weighted_kl_map)
    save_to_pkl(wt_kl_maps, os.path.join(outfile_path, 'wt_kl_maps.pkl'))

    if args.compute_statistics:
        out_stats_file.write('\n\nMethod: Weighted KL Divergence (using {d} similar (SNOMED-CT) terms) \n')
        for k in K:
            correct, not_found, score = compute_statistics(
                            src_df=source_df,
                            tgt_df=target_df,
                            maps=weighted_kl_map,
                            gt_dict=gt_dict,
                            k = k,
                            gt_scores=gt_scores
                        )

            accuracy = round(correct / (len(source_df) - not_found), 4)
            hs_score = round(score / (len(source_df) - not_found), 4)
            out_stats_file.write(f'Top-{k} accuracy: \t {accuracy}, [{correct} | {hs_score} | {not_found} | {len(source_df)}] \n')

    if args.compute_statistics:
        out_stats_file.close()