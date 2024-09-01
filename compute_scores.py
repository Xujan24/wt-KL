import pathlib
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.helpers import save_to_pkl, hierarchical_similarity_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    ## load dataset
    df = pd.read_csv(args.data_path, dtype=str)
    codes = df['code'].to_list()
    n = len(codes)
    
    del df

    ## compute scores
    scores = np.identity(n)
    tqdm.write(f'Computing the scores for {pathlib.Path(args.data_path).stem} ...')
    for i in tqdm(range(len(codes))):
        a = codes[i]
        for j in range(i+1, n):
            b = codes[j]
            scores[i][j] = hierarchical_similarity_score(a, b)
            scores[j][i] = scores[i][j]

    ## save scores
    save_to_pkl(scores, args.out_path)