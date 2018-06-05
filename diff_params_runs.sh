#!/bin/bash

# Run with different vector size for "Deliverable 2"
# python main.py -lr 0.8 -bs 50 -niter 20000 -ws 5 -vs 10 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
# python main.py -lr 0.8 -bs 50 -niter 20000 -ws 5 -vs 75 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
# python main.py -lr 0.8 -bs 50 -niter 20000 -ws 5 -vs 150 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
# python main.py -lr 0.8 -bs 50 -niter 20000 -ws 5 -vs 225 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
# python main.py -lr 0.8 -bs 50 -niter 20000 -ws 5 -vs 300 -ns 10 -lrd 20000 -alpha 0.01 -vi 400

# Run with different learning rates
# python main.py -lr 3.0 -bs 50 -niter 20000 -ws 5 -vs 75 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
# python main.py -lr 2.0 -bs 50 -niter 20000 -ws 5 -vs 75 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
# python main.py -lr 1.0 -bs 50 -niter 20000 -ws 5 -vs 75 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
#python main.py -lr 4.0 -bs 50 -niter 20000 -ws 5 -vs 75 -ns 10 -lrd 20000 -alpha 0.01 -vi 400
python main.py -lr 5.0 -bs 50 -niter 20000 -ws 5 -vs 75 -ns 10 -lrd 20000 -alpha 0.01 -vi 400

# Run model with 2 dimension to create an embedding for deliverable 4
python main.py -lr 0.8 -bs 50 -niter 20000 -ws 5 -vs 2 -ns 10 -lrd 10000 -alpha 0.01 -vi 400
