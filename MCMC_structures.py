import numpy as np
import matplotlib.pyplot as plt
import corner
from emcee.autocorr import integrated_time



# class structure for model
class Model:
    
    def __init__(self, num_params, param_mins, param_maxs, param_labels, in_domain_func, lnlike_func, lnprior_func, get_fisher_func=None):
        self.num_params = num_params
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.param_labels = param_labels
        self.in_domain_func = in_domain_func
        self.lnlike_func = lnlike_func
        self.lnprior_func = lnprior_func
        self.get_fisher_func = get_fisher_func
        
        
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
        if self.get_fisher_func is not None:
            return self.get_fisher_func(params)
        else:
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
    
    def __init__(self, coordinates, lnpost_val, samples, lnpost_vals, history, temperature, fisher_vals, fisher_vecs, 
                 accept_counts, reject_counts, FIM_jump_selects, FIM_weights, DE_weights, DE_choices1, 
                 DE_choices2, rands_4_acceptance):
        self.coordinates = coordinates
        self.lnpost_val = lnpost_val
        self.samples = samples
        self.lnpost_vals = lnpost_vals
        self.history = history
        self.temperature = temperature
        self.fisher_vals = fisher_vals
        self.fisher_vecs = fisher_vecs
        self.accept_counts = accept_counts
        self.reject_counts = reject_counts
        self.FIM_jump_selects = FIM_jump_selects
        self.FIM_weights = FIM_weights
        self.DE_weights = DE_weights
        self.DE_choices1 = DE_choices1
        self.DE_choices2 = DE_choices2
        self.rands_4_acceptance = rands_4_acceptance
        
        
# class structure for MCMC
class MCMC:
    
    def __init__(self, model, num_samples, jump_blend, num_chains, len_history=1000):
        self.model = model
        self.num_samples = num_samples
        self.jump_blend = jump_blend
        self.num_chains = num_chains
        self.len_history = len_history
        
        # randomly choose jump choices
        self.jump_choices = np.random.choice(range(3), size=self.num_samples, p=jump_blend)
        self.chain_swap_choices = np.random.choice(range(10), size=self.num_samples)
        
        # track parallel-tempering swap acceptance fraction
        self.chain_swap_accept_count = 0
        self.chain_swap_reject_count = 0
        
        
    # initialize chains for MCMC
    def init_chains(self, init_sample=None):
        
        # define temperature ladder
        c = 1.5
        temps = np.array([c**k for k in range(self.num_chains)])
        
        chains = []
        for j in range(self.num_chains):
            samples = np.zeros((self.num_samples, self.model.num_params))
            ln_posteriors = np.zeros(self.num_samples)
            if init_sample is None:
                samples[0] = self.model.get_draws_in_domain(1)
            else:
                samples[0] = init_sample
            ln_posteriors[0] = self.model.eval_lnposterior(samples[0], temps[j])
            history = self.model.get_draws_in_domain(self.len_history)
            fisher_vals, fisher_vecs = np.linalg.eig(self.model.get_fisher(samples[0], temps[j]))
            accept_counts = np.zeros(len(self.jump_blend))
            reject_counts = np.zeros(len(self.jump_blend))
            FIM_jump_selects = np.random.choice(self.model.num_params, size=self.num_samples)
            FIM_weights = np.random.normal(loc=0., scale=1., size=self.num_samples)
            DE_weights = np.random.normal(loc=0., scale=2.38/np.sqrt(2*self.model.num_params), size=self.num_samples)
            DE_choices1 = np.random.choice(range(self.len_history), size=self.num_samples)
            DE_choices2 = np.random.choice(range(self.len_history), size=self.num_samples)
            rands_4_acceptance = np.random.uniform(0., 1., size=self.num_samples)
            chains.append(Chain(samples[0].copy(), ln_posteriors[0].copy(), samples, ln_posteriors, history, temps[j], 
                                fisher_vals, fisher_vecs, accept_counts, reject_counts, FIM_jump_selects,
                                FIM_weights, DE_weights, DE_choices1, DE_choices2, rands_4_acceptance))
    
        return chains
        
        
    
    # update chain using Fisher information jump
    def do_Fisher_jump(self, chains, iteration):
        for j in range(self.num_chains):
            chain = chains[j]
            Fisher_weight = 1 / np.sqrt(abs(chain.fisher_vals[chain.FIM_jump_selects[iteration]])) * chain.FIM_weights[iteration]
            jump = np.real(Fisher_weight * chain.fisher_vecs[:,chain.FIM_jump_selects[iteration]])
            # if jump is NaN propose Gaussian random jump
            # (jump may be NaN because of Fisher matrix)
            if np.isnan(jump).any():
                print('JUMP IS NAN')
                jump = np.random.normal(0, 1, self.model.num_params)
            # update chain coordinates and ln(posterior) value
            chain.coordinates += jump
            chain.lnpost_val = self.model.eval_lnposterior(chain.coordinates, chain.temperature)
        return
    
    # update chain using differential evolution jump
    def do_DE_jump(self, chains, iteration):
        for j in range(self.num_chains):
            chain = chains[j]
            jump = np.real((chain.history[chain.DE_choices1[iteration]] 
                            - chain.history[chain.DE_choices2[iteration]]) * chain.DE_weights[iteration])
            # update chain coordinates and ln(posterior) value
            chain.coordinates += jump
            chain.lnpost_val = self.model.eval_lnposterior(chain.coordinates, chain.temperature)
        return
    
    # update chain using Lorentzian proposal
    def do_Lorentzian_jump(self, chains):
        for j in range(self.num_chains):
            r2=2.
            while(r2>1):
                R = np.random.random(size=self.model.num_params)
                r2 = np.sum(R**2)
            mhat = R/np.sqrt(r2)
            jump = 0.5 * np.tan(np.pi*(np.random.random()-0.5)) * mhat
            # update chain coordinates and ln(posterior) value
            chains[j].coordinates += jump
            chains[j].lnpost_val = self.model.eval_lnposterior(chains[j].coordinates, chains[j].temperature)
        return
            
    
    # do MCMC jump
    def do_MCMC_jump(self, chains, iteration):
        jump_choice = self.jump_choices[iteration]
        if jump_choice == 0:  # Fisher jump
            self.do_Fisher_jump(chains, iteration)
        if jump_choice == 1:  # diffential evolution
            self.do_DE_jump(chains, iteration)
        if jump_choice == 2:  # Lorentzian jump
            self.do_Lorentzian_jump(chains)
        return
         
    # do MCMC
    def get_chains(self, init_sample=None):
        
        # initialize chains
        chains = self.init_chains(init_sample=init_sample)
        
        # main MCMC loop
        for i in range(self.num_samples - 1):
            
            # update progress
            if i % (self.num_samples / 10) == 0:
                print(i)
                
            # update Fisher matrix occasionally
            if i % 100 == 0:
                for j in range(self.num_chains):
                    fisher = self.model.get_fisher(chains[j].coordinates, chains[j].temperature)
                    # sometimes FIM has NaNs because numerical derivatives go outside domain
                    # if this is the case don't update the eigenvalues and eigenvectors
                    if not np.isnan(fisher).any():
                        chains[j].fisher_vals, chains[j].fisher_vecs = np.linalg.eig(fisher)
            
            # jump proposal
            # this updates the chains' coordinates and lnlike_val
            MCMC.do_MCMC_jump(self, chains, i)

            # accept or reject jump proposal     
            for j in range(self.num_chains):      
            
                # calculate acceptance ratio
                acc_ratio = np.exp(chains[j].lnpost_val - chains[j].lnpost_vals[i])
            
                # accept or reject jump proposal
                if chains[j].rands_4_acceptance[i] < acc_ratio: # accept
                    chains[j].accept_counts[self.jump_choices[i]] += 1
                    chains[j].samples[i+1,:] = chains[j].history[i%self.len_history] = chains[j].coordinates.copy()
                    chains[j].lnpost_vals[i+1] = chains[j].lnpost_val.copy()
                else: # reject
                    chains[j].reject_counts[self.jump_choices[i]] += 1
                    chains[j].samples[i+1,:] = chains[j].coordinates = chains[j].samples[i,:].copy()
                    chains[j].lnpost_vals[i+1] = chains[j].lnpost_val = chains[j].lnpost_vals[i].copy()
            

            # ocassionally swap coordinates of chains at different temperatures
            if self.num_chains > 1 and self.chain_swap_choices[i] == 0:
                num_swaps = self.num_chains - 1
                for k in range(num_swaps):
                    # randomly select index to try swap with index + 1
                    ind = np.random.choice(self.num_chains - 1)
                    # swap_prob = (self.model.eval_lnposterior(chains[ind+1].coordinates,chains[ind].temperature)*self.model.eval_lnposterior(chains[ind].coordinates,chains[ind+1].temperature)) / (self.model.eval_lnposterior(chains[ind].coordinates,chains[ind].temperature)*self.model.eval_lnposterior(chains[ind+1].coordinates,chains[ind+1].temperature))
                    swap_prob = np.exp(self.model.eval_lnposterior(chains[ind+1].coordinates,chains[ind].temperature) + self.model.eval_lnposterior(chains[ind].coordinates,chains[ind+1].temperature) - self.model.eval_lnposterior(chains[ind].coordinates,chains[ind].temperature) - self.model.eval_lnposterior(chains[ind+1].coordinates,chains[ind+1].temperature))
                    if np.random.uniform() < swap_prob: # accept swap
                        self.chain_swap_accept_count += 1
                        store_coordinates = chains[ind].coordinates.copy()
                        store_lnposterior = chains[ind].lnpost_val.copy()
                        chains[ind].coordinates = chains[ind+1].coordinates.copy()
                        chains[ind].lnpost_val = chains[ind].lnpost_vals[i+1] = chains[ind+1].lnpost_val.copy() * chains[ind+1].temperature / chains[ind].temperature
                        chains[ind+1].coordinates = store_coordinates
                        chains[ind+1].lnpost_val = store_lnposterior * chains[ind].temperature / chains[ind+1].temperature
                    else:
                        self.chain_swap_reject_count += 1
        return chains



# class structure for post-processing
class PostProcessing:
    
    def __init__(self, chains, mcmc, params_injs):
        self.chains = chains
        self.mcmc = mcmc
        self.params_injs = params_injs
        
        # colors and labels for plotting
        self.param_labels = self.mcmc.model.param_labels
        self.num_params = self.mcmc.model.num_params
        self.temp_colors = plt.get_cmap('jet')(np.linspace(0.1, 0.9, self.mcmc.num_chains))
        self.temp_labels = [str(chain.temperature) for chain in self.chains]
        self.param_colors = plt.get_cmap('jet')(np.linspace(0.1, 0.9, self.num_params))
        
        # set burn-in length
        self.burnin = int(mcmc.num_samples / 10)
        
    # get acceptance fractions
    def get_acc_frac(self):
        acc_fracs = []
        for j in range(self.mcmc.num_chains):
            acc_frac = self.chains[j].accept_counts / (self.chains[j].accept_counts + self.chains[j].reject_counts)
            acc_fracs.append(acc_frac)
        return acc_fracs
    
    # get parallel-tempering chain swap acceptance fraction
    def get_chain_swap_frac(self):
        return (self.mcmc.chain_swap_accept_count) / (self.mcmc.chain_swap_accept_count + self.mcmc.chain_swap_reject_count)
    
    # estimate maximum a posteriori (MAP)
    def get_MAP(self, chain_ind=0):
        ln_posterior_vals = self.chains[chain_ind].lnpost_vals
        MAP_index = list(ln_posterior_vals).index(max(ln_posterior_vals))
        return self.chains[chain_ind].samples[MAP_index]
    
    # make trace plot
    def plt_trace(self, chain_ind=0):
        if self.params_injs != None:
            for i in range(self.num_params):
                plt.plot(self.chains[chain_ind].samples[:,i], color=self.param_colors[i], alpha=0.5, label=self.param_labels[i])
                for params in self.params_injs:
                    plt.axhline(params[i], alpha=0.7, color=self.param_colors[i])
        else:
            for i in range(self.num_params):
                plt.plot(self.chains[chain_ind].samples[:,i], color=self.param_colors[i], alpha=0.5, label=self.param_labels[i])
        plt.xlabel('MCMC iteration')
        plt.ylabel('parameter value')
        plt.legend(loc='lower right')
        plt.show()
        return
        
    # plot ln(posterior) samples
    def plt_lnlikes(self):
        for i in range(self.mcmc.num_chains):
            plt.plot(self.chains[i].lnpost_vals, color=self.temp_colors[i], label=self.temp_labels[i], alpha=0.5)
        plt.xlabel('MCMC iteration')
        plt.ylabel('log(posterior)')
        plt.legend(loc='upper right')
        plt.show()
        return
        
    # plot corner plot
    def plt_corner(self, chain_ind=0):
        NP = self.num_params
        fig = corner.corner(self.chains[chain_ind].samples[self.burnin:], labels=self.param_labels, range=[0.99]*NP)
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
        plt.show()
        return
        
        
    # get auto-correlation length of each parameter in samples
    def get_auto_corr(self, chain_ind=0):
        samples = self.chains[chain_ind].samples[self.burnin:]
        auto_corr_length = {param_label: 0 for param_label in self.param_labels}
        for i in range(self.num_params):
            auto_corr_length[self.param_labels[i]] = integrated_time(samples[:,i])[0]
        return auto_corr_length
    
