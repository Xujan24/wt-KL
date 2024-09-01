from typing import Dict, List, Any
from pymedtermino.snomedct import *
import torch
import pickle as pkl
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from ordered_set import OrderedSet
import math

STOPWORDS = list(set(stopwords.words('english')))


def save_to_pkl(obj: Any, path: str) -> None:
    with open(path, 'wb') as file:
        pkl.dump(obj, file)
    return None


def load_from_pkl(path: str) -> Any:
    with open(path, 'rb') as file:
        obj = pkl.load(file)
    return obj


def print_top_k_accuracy(
        results: Dict[str, List[np.ndarray]],
        target_df: pd.DataFrame,
        gt_dict: Dict[str, List[str]],
        k_s: List[int] = [1, 3, 5, 10], 
        sorted: bool = False,
        asc: bool = True
    ) -> None:

    _results = results.copy()

    ## sort if not already sorted
    if not sorted:
        for c, v in results.items():
            ## for some reason when saving the results, its saved as a list of cupy.ndarray
            ## so using cp.array() to convert it to a cupy.ndarray
            if asc:
                v_sorted = np.argsort(np.array(v))
            else:
                v_sorted = np.flip(np.argsort(np.array(v)))

            m = [target_df.iloc[int(i), 0] for i in v_sorted]
            _results[c] = m
            del v_sorted, m

    for  k in k_s:
        correct = 0
        for c, m in _results.items():
            top_k = m[:k]

            gt = gt_dict.get(c)

            if gt is None:
                continue

            if len(set(top_k) & set(gt)) > 0:
                correct += 1

        print(f"   ===Accuracy (Top-{k}): {round(correct/len(gt_dict) * 100, 2)}% ({correct} out of {len(gt_dict)})")

def get_embeddings(model, inputs, device):
    with torch.no_grad():
        out = model.encode(inputs, device=device, convert_to_numpy=True)

    if len(inputs) > 1:
        mu = np.mean(out, axis=0)
        cov = np.cov(out, rowvar=False)
    
    else:
        mu = out[0]
        cov = np.identity(out.shape[1])
    
    return out, mu, np.diag(cov)


def gen_paraphrases(
    tokenizer,
    model,
    device,
    inputs,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128,
) -> List[str]:
    """
    Generates paraphrases for given inputs
    source: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
    """
    input_ids = tokenizer(
        f'paraphrase: {inputs}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = [x.strip().lower() for x in res]

    return res

def get_similar_terms(x: str, model, tokenizer, device, n: int = 15) -> List[str]:
    """Fetches similar terms using the code description alone.
    """
    similar_terms = []
    similar_snomed_concepts = [concepts.term.lower().replace('(disorder)', '').strip() for concepts in SNOMEDCT.search(f'{x} (disorder)')]
    similar_snomed_concepts = list(set(similar_snomed_concepts))
    similar_terms = [x, *similar_snomed_concepts]
    similar_terms = list(set(similar_terms))

    del similar_snomed_concepts

    if len(similar_terms) < n:
        n_req = n - len(similar_terms)
        paraphrases = gen_paraphrases(tokenizer, model, device, x, num_return_sequences=n_req, num_beams=max(n_req, 2), num_beam_groups=max(n_req, 2))
        similar_terms = [*similar_terms, *paraphrases]
        del paraphrases

    return similar_terms




def get_similar_terms_using_hierarchy(x: List[str], model, tokenizer, device, n: int = 15, depth: int = 5) -> List[str]:
    """Fetches similar terms like `get_similar_terms()` only difference being the use of hierarchical information.
    """
    ## format x
    ## remove any nan value
    x = list(filter(lambda a: str(a).lower() != 'nan', x))
    ## remove the code
    x = x[1:]
    ## remove the last item
    x = x[:-1]

    similar_terms = __get_similar_terms_using_snomedct(x[0], depth)

    similar_terms = [*x, *similar_terms]
    similar_terms = list(set(similar_terms))

    if len(similar_terms) < n:
        n_req = n - len(similar_terms)
        paraphrases = gen_paraphrases(tokenizer, model, device, x, num_return_sequences=n_req, num_beams=max(n_req, 2), num_beam_groups=max(n_req, 2))
        similar_terms = [*similar_terms, *paraphrases]
        del paraphrases

    return similar_terms


def hierarchical_similarity_score(a: str, b: str) -> float:
    '''
    Given two code, this function computes the hierarchical similarity score between them.
    '''
    ## we will append these special characters to each corresponding items in a and b
    s = ['#', '##', '###', '####', '#####', '######', '#######']

    ## we use OrdereSet to preserve the order of the elements so that intersection has meaningful interpretation.
    a = OrderedSet([x+y for x, y in zip(list(a), s)])
    b = OrderedSet([x+y for x, y in zip(list(b), s)])

    ## get the number of intersecting items (sequentially from left to right)
    ## we add 1 to ensure there's at least 1 intersecting element between a and b, i.e. the root.
    n = __compute_longest_items_sequentially(a, b) + 1
    m = len(a) + len(b) - n + 2

    r = max(m-n - 1, 0)
    
    return round(math.exp(-r) * n/m, 4)


def __get_similar_terms_using_snomedct(x: str, k: int) -> List[str]:
    concepts = SNOMEDCT.search(f'{x.lower()} (disorder)')
    similar_terms = []
    if len(concepts) <= 3:
        ## tokenize the word and remove any stop words
        _x = word_tokenize(x)
        _x = [w for w in _x if not w.lower() in STOPWORDS]
        
        ## search for the similar terms using each word tokens
        for w in _x:
            if len(w) <= 3:
                continue
            
            _concepts = SNOMEDCT.search(f'{w.lower()} (disorder)')

            if len(_concepts) == 0:
                continue

            _concepts = _concepts[:k]
            
            for _concept in _concepts:
                similar_terms = [*similar_terms, *SNOMEDCT[_concept.code].terms]

    else:
        concepts = concepts[:k]
        
        for concept in concepts:
            similar_terms = [*similar_terms, *SNOMEDCT[concept.code].terms]
        
    similar_terms = [term.lower() for term in similar_terms]
    similar_terms = [term for term in similar_terms if '(disorder)' not in term]
    similar_terms = list(set(similar_terms))

    return similar_terms


def __compute_longest_items_sequentially(x: OrderedSet, z: OrderedSet) -> int:
    '''
    Given two sets, returns the number of exact items sequentially.
    This helps to determine where in the hierarchy the split has occured.
    if x and z doesn't contain a root node and are completely in different branch then it will return 0.
    e.g. if A = ['0', '0', '1', '0'] and B = ['0', '1', '1', '0'] then it will return 1.
    '''
    x_and_z = x & z
    x_and_z = [''.join(list(m)[1:]) for m in x_and_z]

    ## for each index, check if the numper of '#' matches the current index, 
    ## and sum up to get the number of sequentially same parent node.
    return sum([len(x_and_z[i]) - 1 == i for i in range(len(x_and_z))])
    