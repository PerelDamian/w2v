from data_utils import SentenceSplitter, DataParser
from w2v_model import Word2VecModel, SGDLearningHyperParams, Word2VecHyperParams
import argparse
from matplotlib import pyplot as plt
from time import time
from time import gmtime, strftime
import os
import logging
import evaluation

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

    logs_dir = strftime("logs/%Y-%m-%d %H:%M:%S", gmtime()) + '_lr {} bs {} niter {} ws {} vs {} ns {} lrd {} alpha {}'.format(args.lr, args.batch_size,
                                                                                                                               args.n_iterations, args.window_size,
                                                                                                                               args.vectors_size, args.n_negative_words,
                                                                                                                               args.lr_ietrations_decay, args.noise_dist_alpha)
    os.makedirs(logs_dir)

    sgd_params = SGDLearningHyperParams(args.lr, args.batch_size, args.n_iterations, args.validation_interval)
    w2v_params = Word2VecHyperParams(args.window_size, args.vectors_size, args.n_negative_words,
                                     args.lr_ietrations_decay, args.noise_dist_alpha)

    model = Word2VecModel(data, w2v_params)

    model.LearnParamsUsingSGD(sgd_params, logs_dir=logs_dir, use_test=True)

    fn = logs_dir + '/' + "training_ll" + '.log'
    logging.basicConfig(filename=fn, level=logging.INFO)
    # logging.basicConfig(filename=logs_dir + '/' + 'final_log.log', level=logging.INFO)

    # Save final scores to file

    logging.info('\n\n\n')
    logging.info('Learning Rate: {}'.format(args.lr))
    logging.info('Window Size: {}'.format(args.window_size))
    logging.info('Batch Size: {}'.format(args.batch_size))
    logging.info('Number of iterations: {}'.format(args.n_iterations))
    logging.info('Vectors size: {}'.format(args.vectors_size))
    logging.info('number of negative sampling words: {}'.format(args.n_negative_words))
    logging.info('decay iterations each {} steps'.format(args.lr_ietrations_decay))
    logging.info('Noise distribution alpha'.format(args.noise_dist_alpha))

    logging.info('Training Time: {}, \nFinal Train LL: {}, \nFinal Test LL: {}, \n\n'.format(model.training_scores['training_time'],
                                                                                             model.training_scores['train_ll'][-1],
                                                                                             model.training_scores['test_ll'][-1]))
    logging.info('\n\n\nContext words from input word:')

    logging.info('Top 10 Conetxt words for the input word "good" are: ' + str(evaluation.most_likely_cotext_words(model, 'good', 10)))
    logging.info('Top 10 Conetxt words for the input word "bad" are: ' + str(evaluation.most_likely_cotext_words(model, 'bad', 10)))
    logging.info('Top 10 Conetxt words for the input word "lame" are: ' + str(evaluation.most_likely_cotext_words(model, 'lame', 10)))
    logging.info('Top 10 Conetxt words for the input word "cool" are: ' + str(evaluation.most_likely_cotext_words(model, 'cool', 10)))
    logging.info('Top 10 Conetxt words for the input word "exciting" are: ' + str(evaluation.most_likely_cotext_words(model, 'exciting', 10)))
    logging.info('\n\n\n')

    logging.info('Input word from context words:')
    logging.info('The 10 best competition for the sentence "The movie was surprisingly ______" are:' +
                 str(evaluation.most_likely_input_words(model, ['the', 'movie', 'was', 'surprisingly'], 10)))
    logging.info('The 10 best competition for the sentence "______ was really disappointing" are:' +
                 str(evaluation.most_likely_input_words(model, ['was', 'really', 'disappointing'], 10)))
    logging.info('The 10 best competition for the sentence "Knowing that she _____ was the best part" are:' +
                 str(evaluation.most_likely_input_words(model, ['knowing', 'that', 'she', 'was', 'the', 'best', 'part'], 10)))
    logging.info('\n\n\n')

    logging.info('Best Analogies using input embeddings:')
    logging.info('The 10 best suggestions for "man is to woman as men is to___": ' + str(evaluation.top_k_analogy_solver(model, 'man', 'woman', 'man', 10)))
    logging.info('The 10 best suggestions for "good is to great as bad is to___": ' + str(evaluation.top_k_analogy_solver(model, 'good', 'great', 'bad', 10)))
    logging.info('The 10 best suggestions for "warm is to cold as summer is to___": ' + str(evaluation.top_k_analogy_solver(model, 'warm', 'cold', 'summer', 10)))
    logging.info('Same Analogies using context embeddings:')
    logging.info('The 10 best suggestions for "man is to woman as men is to___": ' + str(evaluation.top_k_analogy_solver(model, 'man', 'woman', 'man', 10, 'context')))
    logging.info('The 10 best suggestions for "good is to great as bad is to___": ' + str(evaluation.top_k_analogy_solver(model, 'good', 'great', 'bad', 10, 'context')))
    logging.info('The 10 best suggestions for "warm is to cold as summer is to___": ' + str(evaluation.top_k_analogy_solver(model, 'warm', 'cold', 'summer', 10, 'context')))

    # Deliverable 1 - save plot of ll of train and test as function of iteration
    plt.figure()
    plt.plot(model.training_scores['iters_ll'], model.training_scores['train_ll'], color='blue', label='train')
    plt.plot(model.training_scores['iters_ll'], model.training_scores['test_ll'], color='red', label='test')
    plt.title('Mean LL as function on iteration')
    plt.xlabel('iteration')
    plt.ylabel('Mean LL')
    plt.legend(loc='best')
    plt.savefig(logs_dir + '/' + 'Train_plot.png')