from typing import Dict
import numpy as np
from sentence_transformers.util import cos_sim


def kl_div(p: Dict, q: Dict):
    k = len(p['mu'])
    term1 = np.sum(np.log(q['cov'])) - np.sum(np.log(p['cov']))
    term2 = np.dot((p['mu'] - q['mu']) ** 2, 1/q['cov'])
    term3 = np.sum(p['cov']/q['cov'])

    return 0.5 * (term1 + term2 + term3 - k)


def wasserstein_dist(Mu: Dict, Nu: Dict):
    term1 = np.linalg.norm(Mu['mu'] - Nu['mu'], ord=2)
    term2 = np.linalg.norm(np.sqrt(Mu['cov']) - np.sqrt(Nu['cov']), ord=2)

    return term1 + term2


def bhattacharyya_dist(p: Dict[str, np.ndarray], q: Dict[str, np.ndarray]):
    sigma = (p['cov'] + q['cov'])/2

    term1 = np.dot((p['mu'] - q['mu']) ** 2, 1/sigma)
    term2 = np.sum(np.log(sigma)) - 0.5 * (np.sum(np.log(p['cov'])) + np.sum(np.log(q['cov'])))

    return 1/8 * term1 + 0.5 * term2


def cosine_sim(src: Dict, tgt: Dict):
    src_inputs_mean = [v['mu'] for k, v in src.items()]
    tgt_inputs_mean = [v['mu'] for k, v in tgt.items()]

    sim_score = cos_sim(src_inputs_mean, tgt_inputs_mean).numpy()
    return np.flip(sim_score.argsort(axis=1), axis=1)