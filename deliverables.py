import matplotlib.pyplot as plt


def deliverable_2(vector_sizes, log_likelihoods, training_times):
    plt.figure()
    plt.plot(vector_sizes, log_likelihoods, color='blue', label='train')
    plt.title('Mean LL as function of vector size\nlr 0.8 bs 50 niter 20000 ws 5 ns 10 lrd 20000 alpha 0.01')
    plt.xlabel('vector size')
    plt.ylabel('Mean LL')
    plt.legend(loc='best')
    plt.savefig('logs/deliverable_2_ll_as_vector size.png')

    plt.figure()
    plt.plot(vector_sizes, training_times, color='blue', label='train')
    plt.title('Train Time as function of vector size\nlr 0.8 bs 50 niter 20000 ws 5 ns 10 lrd 20000 alpha 0.01')
    plt.xlabel('vector size')
    plt.ylabel('Train Time')
    plt.legend(loc='best')
    plt.savefig('logs/deliverable_2_train time_as_vector size.png')


def deliverable_3(learning_rates, log_likelihoods, training_times):
    plt.figure()
    plt.plot(learning_rates, log_likelihoods, color='blue', label='train')
    plt.title('Mean LL as function of learning rate\nbs 50 niter 20000 ws 5 vs 75 ns 10 lrd 20000 alpha 0.01')
    plt.xlabel('vector size')
    plt.ylabel('Mean LL')
    plt.legend(loc='best')
    plt.savefig('logs/deliverable_3_ll_as_learning rate.png')

    plt.figure()
    plt.plot(learning_rates, training_times, color='blue', label='train')
    plt.title('Train Time as function of learning rate\nbs 50 niter 20000 ws 5 vs 75 ns 10 lrd 20000 alpha 0.01')
    plt.xlabel('vector size')
    plt.ylim((5000,6000))
    plt.ylabel('Train Time')
    plt.legend(loc='best')
    plt.savefig('logs/deliverable_3_train time_as_lr.png')


if __name__ == '__main__':
    variablle_d_lls = [-9.247586547845055, -9.199273701250165, -9.198547883252363, -9.197149811435777, -9.193113870126744]
    variable_d_training_times = [5268, 5600, 5818, 6057, 6531]
    d = [10, 75, 150, 225, 300]
    deliverable_2(d, variablle_d_lls, variable_d_training_times)

    varable_lr_lls = [-8.777760601665326, -8.814701232499674, -8.864296363903897, -8.965983876187982, -9.14743160137451, -9.199273701250165]
    variable_lr_training_times = [5563, 5561, 5568, 5555, 5584, 5600]
    lrs = [5, 4, 3, 2, 1, 0.8]
    deliverable_3(lrs, varable_lr_lls, variable_lr_training_times)