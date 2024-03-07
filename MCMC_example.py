from MCMC_structures import *
import numpy as np


###################################################################
################## CONSTRUCT MODEL  ###############################
###################################################################


num_params = 2
param_mins = [-5., -5.]
param_maxs = [5., 5.]
param_labels = ['x', 'y']

def in_domain(params):
    in_dom = True
    for i in range(num_params):
        if params[i] < param_mins[i] or params[i] > param_maxs[i]:
            in_dom = False
            continue
    return in_dom

def lnlike_func(params):
    x, y = params
    return np.log(16/(3*np.pi)*(np.exp(-x**2-(9+4*x**2+8*y)**2)+1/2*np.exp(-8*x**2-8*(y-2)**2)))

def lnprior_func(params):
    for k in range(len(params)):
        if params[k] < param_mins[k] or params[k] > param_maxs[k]:
            return -np.inf
    else:
        return 0


model = Model(num_params, param_mins, param_maxs, param_labels, in_domain, lnlike_func, lnprior_func)


###################################################################
######################## DO MCMC ##################################
###################################################################


jump_blend = 5 # jump_blend/10 is fraction of Fisher jumps (otherwise DE)
num_chains = 5
len_history = 1_000
num_samples = int(1e5)

mcmc = MCMC(model, jump_blend, num_chains, len_history)

# do MCMC!
chains = mcmc.get_chains(num_samples)


###################################################################
###################### POST-PROCESSING ############################
###################################################################

params_injs = [[0., -9./8.], [0., 2.]]
pp = PostProcessing(model, chains, mcmc, params_injs)

print(pp.get_acc_frac())
pp.plt_trace()
pp.plt_lnlikes()

burnin = int(num_samples / 10)
fig = pp.plt_corner(burnin)
fig.savefig('corner.png')