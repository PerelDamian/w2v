import numpy as np
from utils import l2_normalize
from utils import get_noise_dist, sigmoid, ll_prob, choice_k
from data_utils import get_data_pairs
import logging
from time import gmtime, strftime
import pprint
from time import time


class Word2VecHyperParams:
    def __init__(self, window_size, vectors_size, n_negative_words, lr_iterations_decay, noise_dist_alpha, seed=0):
        self.window_size = window_size
        self.vectors_size = vectors_size
        self.n_negative_words = n_negative_words
        self.lr_iterations_decay = lr_iterations_decay
        self.noise_dist_alpha = noise_dist_alpha
        self.seed = seed


class SGDLearningHyperParams:
    def __init__(self, lr, batch_size, n_iterations, validation_interval):
        self.lr = lr
        self.batch_size = batch_size
        self.validation_interval = validation_interval
        self.n_iterations = n_iterations


class Word2VecModel:
    def __init__(self, data, params):

        self.params = params

        self.data = data

        self.noise_dist = get_noise_dist(self.data.train, params.noise_dist_alpha)

        self.u = np.random.normal(loc=0.0, scale=0.01, size=(self.data.voc_size, params.vectors_size))
        self.u = l2_normalize(self.u, axis=1)

        self.v = np.random.normal(loc=0.0, scale=0.01, size=(self.data.voc_size, params.vectors_size))
        self.v = l2_normalize(self.v, axis=1)

        self.training_scores = {'training_time': 0,
                                'test_ll': [],
                                'train_ll': [],
                                'iters_ll': []
                                }

    def get_gradient(self, context_ind, input_ind, k_neg_sam_ind):
        # ToDo - Check
        Vc = self.v[context_ind]
        Ui = self.u[input_ind]
        neg_vecs = self.v[k_neg_sam_ind]
        K = neg_vecs.shape[0]

        # grad by Ui
        dUi = (1 - sigmoid(np.sum(Vc * Ui))) * Vc
        dUi -= np.sum(sigmoid(np.sum(Ui * neg_vecs, axis=1))[:, None] * neg_vecs, axis=0) / K

        # grad by Vc
        dVc = (1 - sigmoid(np.sum(Vc * Ui))) * Ui

        # grad by neg vecs
        dV_neg = sigmoid(np.sum(Ui * neg_vecs, axis=1))[:, None] * Ui / K

        return dUi, dVc, dV_neg

    def sgd_update(self, batch_data, learning_params):
        # the updates are done for each pair. this way we can avoid creating matrices with the vocabulary size dimension
        per_pair_lr = learning_params.lr / learning_params.batch_size

        for input_ind, context_ind in batch_data:
            k_neg_sam_ind = choice_k(self.noise_dist, self.params.n_negative_words)
            pair_du, pair_dv_cntxt_word, pair_dv_neg_words = self.get_gradient(context_ind, input_ind, k_neg_sam_ind)

            self.u[input_ind] += pair_du * per_pair_lr
            self.v[context_ind] += pair_dv_cntxt_word * per_pair_lr
            self.v[k_neg_sam_ind] += pair_dv_neg_words * per_pair_lr

            self.u[input_ind] = l2_normalize(self.u[input_ind], axis=1)
            self.v[context_ind] = l2_normalize(self.v[context_ind], axis=1)
            self.v[k_neg_sam_ind] = l2_normalize(self.v[k_neg_sam_ind], axis=1)

    def LearnParamsUsingSGD(self, learning_params, use_test=True):
        fn = strftime("logs/%Y-%m-%d %H:%M:%S", gmtime()) + '_' + pprint.pformat(learning_params) + '.log'
        logging.basicConfig(filename=fn, level=logging.INFO)
        logging.info('ll')
        logging.info('ll')

        begin_time = time()
        for iter_ind in np.arange(learning_params.n_iterations):
            if iter_ind % self.params.lr_iterations_decay == 0 and iter_ind != 0:
                learning_params.lr /= 2

            batch_data = self.data.get_batch_data(learning_params.batch_size, self.params.window_size)
            self.sgd_update(batch_data, learning_params)
            batch_mll = self.batch_mean_ll(batch_data)
            print('{}. Batch mean Log-Likelihood is: {}'.format(iter_ind, batch_mll))

            if iter_ind % learning_params.validation_interval == 0 and use_test:
                test_mean_ll = self.data_mean_ll(self.data.test, max_pairs=20000)
                train_mean_ll = self.data_mean_ll(self.data.train, max_pairs=20000)
                print('{}. Batch mean LL: {}, Test mean LL: {}'.format(iter_ind, train_mean_ll, test_mean_ll))
                logging.info('{}. Batch mean LL: {}, Test mean LL: {}'.format(iter_ind, train_mean_ll, test_mean_ll))
                self.training_scores['test_ll'].append(test_mean_ll)
                self.training_scores['train_ll'].append(train_mean_ll)
                self.training_scores['iters_ll'].append(iter_ind)

        self.training_scores['training_time'] = time() - begin_time

    def data_mean_ll(self, sentences_lst, max_pairs):
        pairs = get_data_pairs(sentences_lst, self.params.window_size, max_pairs=max_pairs)
        return self.batch_mean_ll(pairs)

    def batch_mean_ll(self, pairs):
        lls = [ll_prob(self.u[inp_ind], self.v[cntxt_ind], self.v) for inp_ind, cntxt_ind in pairs]
        return np.mean(lls)
