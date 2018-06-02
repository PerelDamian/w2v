import numpy as np
from matplotlib import pyplot as plt


def top_k_words_from_ll_values(model, lst, k):
    top_k_indexes = np.argsort(lst)[-k-1:]

    top_k_words = [model.data.ind2word[ind] for ind in top_k_indexes[1:] if ind != 0]

    if len(top_k_words) < k:  # case <unk> is one of the top words
        top_k_words.append(model.data.ind2word[top_k_indexes[0]])

    return top_k_words


def most_likely_cotext_words(model, input_word, k):
    input_word_ind = model.data.word2ind.get(input_word, 0)

    Ui = model.u[input_word_ind]

    likelihood = np.sum(model.v * Ui, axis=1)

    return top_k_words_from_ll_values(model, likelihood, k)


def most_likely_input_words(model, context_words, k):
    likelihood = np.zeros(model.u.shape[0])

    for i, Ui in enumerate(model.u):
        for context_word in context_words:
            context_word_ind = model.data.word2ind.get(context_word, 0)

            Vc = model.v[context_word_ind]

            ll_nominators = np.exp(np.sum(Ui * Vc))
            ll_denoninators = np.sum(np.exp(Ui.dot(model.v.T)))

            likelihood[i] += np.log(ll_nominators / ll_denoninators)

    return top_k_words_from_ll_values(model, likelihood, k)


def top_k_analogy_solver(model, a, b, c, k, emb='input'):
    """
    returns top k i from argmax(Ui * (Ua-Ub+Uc))
    example a=man, b=woman, c=king. expected to return queen
    """
    a_ind = model.data.word2ind.get(a, 0)
    b_ind = model.data.word2ind.get(b, 0)
    c_ind = model.data.word2ind.get(c, 0)

    if emb == 'context':
        mat = model.v
    else:
        mat = model.u

    expected_vec = mat[a_ind] + mat[b_ind] - mat[c_ind]

    likelihood = np.sum(mat * expected_vec, axis=1)

    return top_k_words_from_ll_values(model, likelihood, k)


def visualize_words_first_2_components(model, words, filename):
    words_inds = [model.data.word2ind.get(word, 0) for word in words]

    words_vecs = model.u[words_inds]

    words_vecs = words_vecs[:, :2]

    plt.figure()
    plt.scatter(words_vecs[:, 0], words_vecs[:, 1])
    for i, word in enumerate(words):
        plt.text(words_vecs[i, 0], words_vecs[i, 1], word)
    plt.savefig(r'visualizations/' + filename)




