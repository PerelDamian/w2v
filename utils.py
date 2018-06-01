import numpy as np
from collections import Counter


def l2_normalize(v, axis=1):
    if len(v.shape) == 1:
        norm = np.linalg.norm(v)
        return v / norm
    else:
        norm = np.linalg.norm(v, axis=axis)
    # if norm == 0:
    #     return v
        return v / norm[:, None]


def get_noise_dist(data, alpha):
    unigram_counts = dict(Counter([word for sntnce in data for word in sntnce]))

    for word in unigram_counts:
        unigram_counts[word] = unigram_counts[word] ** alpha

    sum_unnorm_prob = sum(unigram_counts.values())
    noise_dist = {}
    for word in unigram_counts:
        noise_dist[word] = unigram_counts[word] / sum_unnorm_prob

    return noise_dist


def choice_k(dist, k):
    return np.random.choice(list(dist.keys()), size=k, replace=True, p=list(dist.values()))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def neg_sampling_ll_prob(input_vec, context_vec, neg_sampl_matrix):
    positive_part = np.log(sigmoid(np.sum(input_vec * context_vec)))
    negative_part = np.sum(np.log(1 - sigmoid(np.sum(input_vec * neg_sampl_matrix, axis=1)))) / neg_sampl_matrix.shape[0]

    return positive_part + negative_part


def ll_prob(Ui, Vc, V):
    nomi = np.exp(np.sum(Ui * Vc))
    den = np.sum(np.exp(Ui.dot(V.T)))

    return np.log(nomi / den)
