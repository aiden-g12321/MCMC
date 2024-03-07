import numpy as np
import matplotlib.pyplot as plt
import corner



# class structure for parameters
class Model:
    
    def __init__(self, num_params, param_mins, param_maxs, param_labels, in_domain_func, lnlike_func, lnprior_func):
        self.num_params = num_params
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.param_labels = param_labels
        self.in_domain_func = in_domain_func
        self.lnlike_func = lnlike_func
        self.lnprior_func = lnprior_func
        
    # evaluate ln(likelihood) for given parameter values and temperature
    def eval_lnlike(self, params, temperature):
        return (1 / temperature) * self.lnlike_func(params)
    
    # evaluate ln(prior) for given parameter values and temperature
    def eval_lnprior(self, params, temperature):
        return (1 / temperature) * self.lnprior_func(params)
    
    # evaluate ln(posterior) for given parameter values and temperature
    def eval_lnposterior(self, params, temperature):
        if not self.in_domain_func(params):
            return -np.inf
        else:
            return self.eval_lnlike(params, temperature) + self.eval_lnprior(params, temperature)
    
    # small step size to take partial derivative of likelihood
    step_size = 1.e-4
    # compute partial derivative of ln(likelihood)
    def partial_lnposterior(self, params, index, temperature):
        dstep = np.zeros(self.num_params)
        dstep[index] = self.step_size / 100.
        lnposterior1 = self.eval_lnposterior(params - dstep, temperature)
        lnposterior2 = self.eval_lnposterior(params + dstep, temperature)
        return (lnposterior2 - lnposterior1) / (2 * dstep[index])

    # get Fisher information matrix
    def get_fisher(self, params, temperature):
        NP = self.num_params
        FIM = np.zeros((self.num_params, self.num_params))
        for j in range(self.num_params):
            dstep1 = np.zeros(self.num_params)
            dstep1[j] = Model.step_size
            for k in range(j, NP):
                FIM[j,k] = FIM[k,j] = -(self.partial_lnposterior(params + dstep1, k, temperature) 
                                              - self.partial_lnposterior(params - dstep1, k, temperature)) / (2 * Model.step_size)
        return FIM
    
    # get inverse FIM with SVD
    def get_inv_fisher(self, params, temperature):
        return np.linalg.pinv(self.get_fisher(params, temperature))
    
    # draw from uniform distribution in domain
    def get_draws_in_domain(self, num_draws):
        if num_draws == 1:
            draw = np.array([np.random.uniform(self.param_mins[k], self.param_maxs[k]) for k in range(self.num_params)])
            while not self.in_domain_func(draw):
                draw = np.array([np.random.uniform(self.param_mins[k], self.param_maxs[k]) for k in range(self.num_params)])
            return draw
        else:
            draws = []
            for i in range(num_draws):
                draws.append(self.get_draws_in_domain(1))
            return np.array(draws)
                
    
    
# class structure for a chain used in MCMCs
class Chain:
    
    def __init__(self, coordinates, posterior_val, samples, posterior_vals, history, temperature, fisher_vals, fisher_vecs, accept_count, reject_count):
        self.coordinates = coordinates
        self.posterior_val = posterior_val
        self.samples = samples
        self.posterior_vals = posterior_vals
        self.history = history
        self.temperature = temperature
        self.fisher_vals = fisher_vals
        self.fisher_vecs = fisher_vecs
        self.accept_count = accept_count
        self.reject_count = reject_count
        
        
        
# class structure for MCMC
class MCMC:
    
    def __init__(self, model, jump_blend, num_chains, len_history):
        self.model = model
        self.jump_blend = jump_blend
        self.num_chains = num_chains
        self.len_history = len_history
    
    # update chain coordinates and lnlike_vals
    def do_MCMC_jump(self, chains):
        
        # update each chain
        for j in range(self.num_chains):
            # Fisher jump or differential evolution
            jump_choice = np.random.choice(10)

            if jump_choice < self.jump_blend: # Fisher jump
                jump_select = np.random.choice(self.model.num_params)
                jump = np.real(1 / np.sqrt(abs(chains[j].fisher_vals[jump_select])) * chains[j].fisher_vecs[:,jump_select] * np.random.normal())
            
            else: # differential evolution
                history_index1 = np.random.choice(range(self.len_history))
                history_index2 = np.random.choice(range(self.len_history))
                jump = np.real((chains[j].history[history_index1] 
                                - chains[j].history[history_index2]) * np.random.normal(0, 2.38/np.sqrt(2*self.model.num_params)))
            
            # if jump is NaN propose Gaussian random jump
            # (jump may be NaN because of Fisher matrix)
            if np.isnan(jump).any():
                print('JUMP IS NAN')
                jump = np.random.normal(0, 1, self.model.num_params)
            
            chains[j].coordinates += jump
            chains[j].posterior_val = self.model.eval_lnposterior(chains[j].coordinates, chains[j].temperature)
        
        return
            
            
    # do MCMC
    def get_chains(self, num_samples):
        
        # rename objects
        NP = self.model.num_params
        NC = self.num_chains
        
        # define temperature ladder
        c = 1.5
        temps = np.array([c**k for k in range(NC)])
        
        # instantiate chains
        chains = []
        for j in range(NC):
            samples = np.zeros((num_samples, NP))
            ln_posteriors = np.zeros(num_samples)
            samples[0] = self.model.get_draws_in_domain(1)
            ln_posteriors[0] = self.model.eval_lnposterior(samples[0], temps[j])
            history = self.model.get_draws_in_domain(self.len_history)
            fisher_vals, fisher_vecs = np.linalg.eig(self.model.get_fisher(samples[0], temps[j]))
            accept_count = 0
            reject_count = 0
            chains.append(Chain(samples[0].copy(), ln_posteriors[0].copy(), samples, ln_posteriors, history, temps[j], fisher_vals, fisher_vecs, accept_count, reject_count))
        
        
        # main MCMC loop
        for i in range(num_samples - 1):
            
            # update progress
            if i % (num_samples / 10) == 0:
                print(i)
                
            # update Fisher matrix occasionally
            if i % 100 == 0:
                for j in range(NC):
                    fisher = self.model.get_fisher(chains[j].coordinates, chains[j].temperature)
                    # sometimes FIM has NaNs because numerical derivatives go outside domain
                    # if this is the case don't update the eigenvalues and eigenvectors
                    if not np.isnan(fisher).any():
                        chains[j].fisher_vals, chains[j].fisher_vecs = np.linalg.eig(fisher)
            
            # jump proposal
            # this updates the chains' coordinates and lnlike_val
            MCMC.do_MCMC_jump(self, chains)

            # accept or reject jump proposal     
            for j in range(NC):      
            
                # calculate acceptance ratio
                acc_ratio = np.exp(chains[j].posterior_val - chains[j].posterior_vals[i])
            
                # accept or reject jump proposal
                if np.random.uniform() < acc_ratio: # accept
                    chains[j].accept_count += 1
                    chains[j].samples[i+1,:] = chains[j].coordinates.copy()
                    chains[j].posterior_vals[i+1] = chains[j].posterior_val.copy()
                else: # reject
                    chains[j].reject_count += 1
                    chains[j].samples[i+1,:] = chains[j].coordinates = chains[j].samples[i,:].copy()
                    chains[j].posterior_vals[i+1] = chains[j].posterior_val = chains[j].posterior_vals[i].copy()
            

            # ocassionally swap coordinates of chains at different temperatures
            if np.random.choice(10) == 0:
                num_swaps = NC
                for k in range(num_swaps):
                    # randomly select index to try swap with index + 1
                    ind = np.random.choice(NC - 1)
                    swap_prob = (self.model.eval_lnposterior(chains[ind+1].coordinates,chains[ind].temperature)*self.model.eval_lnposterior(chains[ind].coordinates,chains[ind+1].temperature)) / (self.model.eval_lnposterior(chains[ind].coordinates,chains[ind].temperature)*self.model.eval_lnposterior(chains[ind+1].coordinates,chains[ind+1].temperature))
                    if np.random.uniform() < swap_prob: # accept swap
                        store_coordinates = chains[ind].coordinates.copy()
                        store_lnposterior = chains[ind].posterior_val.copy()
                        chains[ind].coordinates = chains[ind+1].coordinates.copy()
                        chains[ind].posterior_val = chains[ind].posterior_vals[i+1] = chains[ind+1].posterior_val.copy() * chains[ind+1].temperature / chains[ind].temperature
                        chains[ind+1].coordinates = store_coordinates
                        chains[ind+1].posterior_val = store_lnposterior * chains[ind].temperature / chains[ind+1].temperature
        
        return chains



# class structure for post-processing
class PostProcessing:
    
    def __init__(self, model, chains, mcmc, params_injs):
        self.model = model
        self.chains = chains
        self.mcmc = mcmc
        self.params_injs = params_injs
        
        self.temp_colors = plt.get_cmap('jet')(np.linspace(0.1, 0.9, self.mcmc.num_chains))
        self.temp_labels = [str(chain.temperature) for chain in self.chains]
        self.param_colors = plt.get_cmap('jet')(np.linspace(0.1, 0.9, self.model.num_params))
        self.param_labels = self.model.param_labels
        
    # get acceptance fractions
    def get_acc_frac(self):
        acc_fracs = []
        for j in range(self.mcmc.num_chains):
            acc_frac = self.chains[j].accept_count / (self.chains[j].accept_count + self.chains[j].reject_count)
            acc_fracs.append(acc_frac)
        return acc_fracs
    
    # make trace plot
    def plt_trace(self, chain_ind=0):
        
        if self.params_injs != None:
            for i in range(self.model.num_params):
                plt.plot(self.chains[chain_ind].samples[:,i], color=self.param_colors[i], alpha=0.5, label=self.param_labels[i])
                for params in self.params_injs:
                    plt.axhline(params[i], alpha=0.7, color=self.param_colors[i])

        else:
            for i in range(self.model.num_params):
                plt.plot(self.chains[chain_ind].samples[:,i], color=self.param_colors[i], alpha=0.5, label=self.param_labels[i])
       
        plt.xlabel('MCMC iteration')
        plt.ylabel('parameter value')
        plt.legend(loc='lower right')
        plt.show()
        
    # plot ln(posterior) samples
    def plt_lnlikes(self):
        
        for i in range(self.mcmc.num_chains):
            plt.plot(self.chains[i].posterior_vals, color=self.temp_colors[i], label=self.temp_labels[i], alpha=0.5)
        
        plt.xlabel('MCMC iteration')
        plt.ylabel('log(posterior)')
        plt.legend(loc='upper right')
        plt.show()
        
        
    # plot corner plot
    def plt_corner(self, burnin, chain_ind=0):
        
        NP = self.model.num_params
        fig = corner.corner(self.chains[chain_ind].samples[burnin:], labels=self.param_labels, range=[0.99]*NP)
        axes = np.array(fig.axes).reshape((NP, NP))
        # Loop over the diagonal
        for i in range(NP):
            ax = axes[i, i]
            for params in self.params_injs:
                ax.axvline(params[i])
        # Loop over the histograms
        for yi in range(NP):
            for xi in range(yi):
                ax = axes[yi, xi]
                for params in self.params_injs:    
                    ax.axvline(params[xi])
                    ax.axhline(params[yi])
                    ax.plot(params[xi], params[yi])
        
        return fig
    
