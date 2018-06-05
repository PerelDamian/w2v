Word Embedding using skip gram and negative samplign

In order to train a model you should run the main.py file with the following flags and args:

	example - python main.py -lr 0.3 -bs 50 -niter 20000 -ws 2 -vs 50 -ns 10 -lrd 3000 -alpha 1 -vi 1000

	lr - learning rate for the SGD
	bs - batch size for the SGD
	niter - number of iterations of the sgd algorithm
	ws - window size to create context\input pairs for the skip gram
	vs - The dimension of the embedding vectors
	ns - number of negative sampling examples for each pair Log Likelihood
	lrd - The number of iterations to make before reducing the lr by 50%.
	alpha - Constant to calculate the unigram distribution for the random sampling of pairs during the training.
	vi - validation interval. each $'vi' iterations, the algorithm will calculate the mean log likelihood on the train and test sets.

Each time the main.py is ran, a directory with all the logs and deliverables is created and saved in the /logs directory.

Deliverables which are created with scores of multiples models, can be created by running the deliverabels.py script with no args.

Running bash diff_params.run.sh will create all needed model needed for the deliverables.

