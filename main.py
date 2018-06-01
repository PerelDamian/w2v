from data_utils import SentenceSplitter, DataParser
from w2v_model import Word2VecModel, SGDLearningHyperParams, Word2VecHyperParams
import argparse
from matplotlib import pyplot as plt
from time import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # example: python main.py -lr 0.3 -bs 50 -niter 20000 -ws 2 -vs 50 -ns 10 -lrd 3000 -alpha 1 -vi 1000
    parser.add_argument("-lr", "--lr", help="Learning Rate", type=float)
    parser.add_argument("-bs", "--batch_size", help="Batch Size", type=int)
    parser.add_argument("-niter", "--n_iterations", help="Number of iterations", type=int)
    parser.add_argument("-ws", "--window_size", help="Window Size", type=int)
    parser.add_argument("-vs", "--vectors_size", help="vectors dimension", type=int)
    parser.add_argument("-ns", "--n_negative_words", help="Negative Sampling count", type=int)
    parser.add_argument("-lrd", "--lr_ietrations_decay", help="lr_ietrations_decay", type=int)
    parser.add_argument("-alpha", "--noise_dist_alpha", help="noise_dist_alpha", type=float)
    parser.add_argument("-vi", "--validation_interval", help="validation_interval", type=int)

    args = parser.parse_args()

    spltr = SentenceSplitter(r'data/datasetSplit.txt')
    data = DataParser(r'data/datasetSentences.txt', spltr)

    for d in [75, 150, 225, 300]:
        beg_tim = time()
        sgd_params = SGDLearningHyperParams(args.lr, args.batch_size, args.n_iterations, args.validation_interval)
        w2v_params = Word2VecHyperParams(args.window_size, d, args.n_negative_words,
                                         args.lr_ietrations_decay, args.noise_dist_alpha)

        model = Word2VecModel(data, w2v_params)

        model.LearnParamsUsingSGD(sgd_params, use_test=True)

        tot_time = (time() - beg_tim) / 60
        plt.figure()
        plt.plot(model.training_scores['iters_ll'], model.training_scores['train_ll'], color='blue', label='train')
        plt.plot(model.training_scores['iters_ll'], model.training_scores['test_ll'], color='red', label='test')
        plt.title('Mean LL as function on iteration - vec dim: ' + str(d))
        plt.xlabel('iteration')
        plt.ylabel('Mean LL')
        plt.legend(loc='best')
        plt.savefig(str(tot_time) + ' minutes dimension ' + str(d)+'.png')


    # todo - Output the hyperparameters and final (mean) log-likelihoods to a file, the file should also
    # todo - include the amount of time needed for training.
