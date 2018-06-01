import numpy as np
from matplotlib import pyplot as plt


def top_k_words_from_ll_values(model, lst, k):
    top_k_indexes = np.argsort(lst)[:k]

    top_k_words = [model.data.ind2word[ind] for ind in top_k_indexes if ind != 0]

    if len(top_k_words) < k:  # case <unk> is one of the top words
        top_k_words.append(model.ind2word_map[top_k_indexes[k]])

    return top_k_words


def most_likely_cotext_words(model, input_word, k):
    input_word_ind = model.data.word2ind.get(input_word, 0)

    Ui = model.u[input_word_ind]

    likelihood = np.sum(model.v * Ui, axis=1)

    return top_k_words_from_ll_values(model, likelihood, k)


def most_likely_input_words(model, context_words, k):
    likelihood = np.zeros(model.u.shape[0])

    ll_denoninators = np.sum(np.exp(model.u.dot(model.v)), axis=1)
    for context_word in context_words:
        context_word_ind = model.data.word2ind.get(context_word, 0)

        Vc = model.v[context_word_ind]

        ll_nominators = np.exp(np.sum(model.u * Vc, axis=1))

        likelihood += np.log(ll_nominators / ll_denoninators)

    return top_k_words_from_ll_values(model, likelihood, k)


def top_k_analogy_solver(model, a, b, c, k):
    """
    returns top k i from argmax(Ui * (Ua-Ub+Uc))
    example a=man, b=woman, c=king. expected to return queen
    """
    a_ind = model.word2ind_map.get(a, 0)
    b_ind = model.word2ind_map.get(b, 0)
    c_ind = model.word2ind_map(c, 0)

    expected_vec = model.u[a_ind] + model.u[b_ind] - model.u[c_ind]

    likelihood = np.sum(model.u * expected_vec, axis=1)

    return top_k_words_from_ll_values(model, likelihood, k)


def visualize_words_first_2_components(model, words, filename):
    words_inds = [model.word2ind_map.get(word, 0) for word in words]

    words_vecs = model.u[words_inds]

    words_vecs = words_vecs[:, :2]

    plt.figure()
    plt.scatter(words_vecs[:, 0], words_vecs[:, 1])
    for i, word in enumerate(words):
        plt.text(words_vecs[i, 0], words_vecs[i, 1], word)
    plt.savefig(r'visualizations/' + filename)




