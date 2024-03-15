from MCMC_structures import *
import numpy as np
from time import time


###################################################################
################## CONSTRUCT MODEL ################################
###################################################################

# define number of parameters, minimum and maximum values, and labels
num_params = 2
param_mins = [-5., -5.]
param_maxs = [5., 5.]
param_labels = ['x', 'y']

# returns true if parameters are in allowed domain, otherwise false
def in_domain(params):
    in_dom = True
    for i in range(num_params):
        if params[i] < param_mins[i] or params[i] > param_maxs[i]:
            in_dom = False
            continue
    return in_dom

# ln(likelihood) function
def lnlike_func(params):
    x, y = params
    return np.log(16/(3*np.pi)*(np.exp(-x**2-(9+4*x**2+8*y)**2)+1/2*np.exp(-8*x**2-8*(y-2)**2)))

# ln(prior) function
def lnprior_func(params):
    for k in range(len(params)):
        if params[k] < param_mins[k] or params[k] > param_maxs[k]:
            return -np.inf
    else:
        return 0

# construct model object
model = Model(num_params, param_mins, param_maxs, param_labels, in_domain, lnlike_func, lnprior_func)


###################################################################
######################## DO MCMC ##################################
###################################################################

# define jump blend, number of chains, length of history, and number of samples to draw
jump_blend = [0.5, 0.5]
num_chains = 5
len_history = 1_000
num_samples = int(1e5)

# construct MCMC object
mcmc = MCMC(model, num_samples, jump_blend, num_chains, len_history)

# do MCMC!
time_start = time()
chains = mcmc.get_chains()
time_stop = time()
duration = time_stop - time_start
print('Completed ' + str(num_samples) + ' MCMC iterations in ' + str(duration) + ' s.')


###################################################################
###################### POST-PROCESSING ############################
###################################################################

params_injs = [[0., -9./8.], [0., 2.]]
burnin = int(num_samples / 10)
pp = PostProcessing(model, chains, mcmc, params_injs)

# print acceptance fractions
print('acceptance fraction per chains:')
print(pp.get_acc_frac())

# print maximum a posteriori (MAP) parameters
print('MAP parameters:')
print(pp.get_MAP())

# make trace plot, plot posterior values, and corner plot
pp.plt_trace()
pp.plt_lnlikes()
pp.plt_corner(burnin)

