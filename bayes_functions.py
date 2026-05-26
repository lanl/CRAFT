"""
Bayesian Fitting Functions for CRAFT
Contains utility functions for MCMC sampling, model loading, and likelihood calculations
Includes both Adaptive MCMC and DREAM samplers
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import joblib
import os
import tensorflow as tf
from typing import Callable, Tuple, List


def log_prob(regrli, scarlerli, x, scaler_pars, Varset2, samples, Model_List, Scaler_List, 
             Obs_List, Obs_save, Varset, Names, Names2, obli):
    """
    Calculate the log probability of a given set of parameters.

    Parameters:
    -----------
    regrli : dict
        Dictionary of regression models
    scarlerli : dict
        Dictionary of scalers for models
    x : array
        The input parameters (transformed/scaled)
    scaler_pars : StandardScaler
        Scaler for parameters
    Varset2 : array
        Variable names (excluding time variables)
    samples : DataFrame
        Sample data for bounds checking
    Model_List : list
        List of model names
    Scaler_List : list
        List of scaler names
    Obs_List : list
        List of observation set names
    Obs_save : DataFrame
        Observation data
    Varset : array
        Full variable set
    Names : Series
        Column names with year, month
    Names2 : Series
        Column names with Year, DOY
    obli : dict
        Dictionary of observation scalers

    Returns:
    --------
    float : The log probability of the input parameters
    """
    # Inverse transform the input parameters
    x = scaler_pars.inverse_transform(x.reshape(1, -1))
   
    outs = pd.DataFrame({}, index=[0])
    for i in list(range(0, len(Varset2))):
        outs[Varset2[i]] = x[0][i]
    x = outs

    # Initialize the output
    DFout = pd.DataFrame({})
    
    # Loop through all our model types
    for i in list(range(0, len(Obs_List))):
        ob_set = Obs_save[Obs_save["Set"] == Obs_List[i]]
        #print(ob_set)
        obscaler = obli[Obs_List[i]]
        #print(Obs_List[i])
        # Load new model/scaler
        if (ob_set["Date"].iloc[0]=="MeanPFT"):
            model_info = regrli[Model_List[0]]  
            regr = model_info['model']
            model_type = model_info['type']
            scaler = scarlerli[Scaler_List[0]]

        else:
            model_info = regrli[Model_List[i]]  
            regr = model_info['model']
            model_type = model_info['type']
            scaler = scarlerli[Scaler_List[i]]
            
     
        # Dynamically check for column
        if (ob_set["Date"].iloc[0]=="MeanPFT"):
            my=ob_set[["Year","pft","case"]].reset_index(drop=True)
            x_1_run_set = pd.concat([pd.concat([x] * len(my)).reset_index(drop=True), my], axis=1)
            one=x_1_run_set[['case','Year','pft', 'fates_rxfire_AB',
                   'p1_fates_leaf_vcmax25top', 'p2_fates_leaf_vcmax25top',
                   'p3_fates_leaf_vcmax25top', 'p4_fates_leaf_vcmax25top',
                   'p3_fates_allom_agb1', 'p1_fates_mort_freezetol',
                   'p2_fates_mort_freezetol', 'p3_fates_mort_freezetol',
                   'p4_fates_mort_freezetol', 'p1_fates_mort_scalar_coldstress',
                   'p2_fates_mort_scalar_coldstress', 'p3_fates_mort_scalar_coldstress',
                   'p4_fates_mort_scalar_coldstress', 'p1_fates_allom_blca_expnt_diff',
                   'p2_fates_allom_blca_expnt_diff', 'p3_fates_allom_blca_expnt_diff',
                   'p4_fates_allom_blca_expnt_diff', 'p1_fates_turnover_fnrt',
                   'p2_fates_turnover_fnrt', 'p4_fates_turnover_fnrt']]
        elif (Obs_List[i] == "LWP_min") | (Obs_List[i] == "LWP_max"):
            my = ob_set[["year", "DOY"]].reset_index(drop=True)
            my["Year"] = my["year"]
            my = my[["year", "DOY"]]
            ob_set = ob_set.sort_values(["year", "DOY"])
            x_1_run_set = pd.concat([pd.concat([x] * len(my)).reset_index(), my], axis=1)
            one = x_1_run_set.iloc[:, 1:].sort_values(["year", "DOY"])
            #one = one.set_axis(Names2, axis=1)
            indices_to_delete = np.where(Varset == "month")[0]
            VarsetDay = np.delete(Varset, indices_to_delete)
            one = one[VarsetDay]
        else: 
            my = ob_set[["year", "month"]].reset_index(drop=True)
            ob_set = ob_set.sort_values(["year", "month"])
            x_1_run_set = pd.concat([pd.concat([x] * len(my)).reset_index(), my], axis=1)
            one = x_1_run_set.iloc[:, 1:].sort_values(["year", "month"])
            one["Year"]=one["year"]
            #print(one)
            indices_to_delete = np.where(Varset == "DOY")[0]
            VarsetMonth = np.delete(Varset, indices_to_delete)
            #print(VarsetMonth)
            one = one[VarsetMonth]
            
        one = scaler.transform(one)
        
        # Make predictions based on model type
        if model_type == "rf":
            predicty = regr.predict(one)
        else:  # Neural network
            # Convert to tensor for TF models
            one_tensor = tf.convert_to_tensor(one.astype('float32'))
            predicty = regr.predict(one_tensor, verbose=0).flatten()
            #print(predicty)
            predicty[predicty < 0] = 0
            
        if (Obs_List[i] == "LWP_min") | (Obs_List[i] == "LWP_max"):
            predicty = np.abs(predicty)
            
        predicty = obscaler.transform(np.log(predicty + 0.0001).reshape(-1, 1))
      
        Frame = pd.concat([ob_set.reset_index(drop=True), 
                          pd.Series(predicty.flatten(), name="sim").reset_index(drop=True)], axis=1)
        if (Obs_List[i] == "LWPmin") | (Obs_List[i] == "LWPmax"):
            Frame = Frame.sample(n=169, random_state=55)
            
        # This is a dataframe with the obs/sim/ in it
        DFout = pd.concat([DFout, Frame])
       # print(Frame)
    # Calculate full log likelihoo
    #print(DFout)
    ll1 = np.sum(norm.logpdf(DFout['sim'], loc=DFout['obs'], scale=DFout['error']*.1))
    
    # Check physical possibility/Values are within the bounds of initial sample
    p = sum([1 for t in range(len(x.columns)) 
             if (x.iloc[0, t] < samples[[x.columns[t]]].min().values) or 
                (x.iloc[0, t] > samples[[x.columns[t]]].max().values)])
    if p >= 1:
        ll1 = -np.inf
        
    return ll1


def boundby(single, min, max):
    """
    Bounce parameter back into bounds if it goes outside
    
    Parameters:
    -----------
    single : float
        Parameter value to check
    min : float
        Minimum bound
    max : float
        Maximum bound
        
    Returns:
    --------
    float : Bounded parameter value
    """
    if single < min:
        single = min + np.abs(min - single)
    if single > max:
        single = max - (single - max)
    if single < min:
        single = min + np.abs(min - single)
    if single > max:
        single = max - (single - max)
    if single > max:
        single = np.random.uniform(min, max)
    if single < min:
        single = np.random.uniform(min, max)
    return single


def bounce(newproposed, samples_scale):
    """
    Bounce all parameters back into bounds
    
    Parameters:
    -----------
    newproposed : array
        Proposed parameter values
    samples_scale : DataFrame
        Scaled samples containing min/max bounds
        
    Returns:
    --------
    array : Bounded proposed parameters
    """
    for i in list(range(0, len(samples_scale.columns))):
        newproposed[i] = boundby(newproposed[i], 
                                samples_scale.iloc[:, i].min(), 
                                samples_scale.iloc[:, i].max())
    return newproposed


def acceptance(x_likelihood, x_new_likelihood):
    """
    Metropolis-Hastings acceptance criterion
    
    Parameters:
    -----------
    x_likelihood : float
        Current log likelihood
    x_new_likelihood : float
        Proposed log likelihood
        
    Returns:
    --------
    bool : True if accept, False if reject
    """
    if x_new_likelihood > x_likelihood:
        return True
    else:
        accept = np.random.uniform(0, 1)
        if accept < np.exp(x_new_likelihood - x_likelihood):
            return True
        else:
            return False


def Loadmodels(LOADIN, EmDir, Model_List, Scaler_List, model_type):
    """
    Load emulator models and scalers
    
    Parameters:
    -----------
    LOADIN : bool
        Whether to load models
    EmDir : str
        Directory containing models
    Model_List : list
        List of model names to load
    Scaler_List : list
        List of scaler names to load
    model_type : str
        Type of model ('rf' or 'nn')
        
    Returns:
    --------
    tuple : (regrli, scarlerli) - dictionaries of models and scalers
    """
    if LOADIN:
        regrli = {}
        scarlerli = {}
        
        for i in list(range(0, len(Model_List))):
            variable_name = Model_List[i]
            # Check if neural network model exists
            nn_model_path = os.path.join(EmDir, variable_name + "_nn.keras")
            rf_model_path = os.path.join(EmDir, variable_name + ".joblib")
            
            model_loaded = False
            
            # Try to load neural network model with .keras extension if it exists
            if model_type == "nn":
                try:
                    print(f"Loading neural network model: {nn_model_path}")
                    model = tf.keras.models.load_model(nn_model_path)
                    regrli[Model_List[i]] = {'model': model, 'type': 'nn'}
                    model_loaded = True
                except Exception as e:
                    print(f"Failed to load neural network model: {e}")
            if model_type == "Xiulin_nn":
                try:
                    print(f"Loading neural network model: {nn_model_path}")
                    model = tf.keras.models.load_model(nn_model_path)
                    regrli[Model_List[i]] = {'model': model, 'type': 'nn'}
                    model_loaded = True
                except Exception as e:
                    print(f"Failed to load neural network model: {e}")
            
            # Try to load Random Forest model if NN didn't work or doesn't exist
            if model_type == "rf":
                try:
                    print(f"Loading Random Forest model: {rf_model_path}")
                    model = joblib.load(rf_model_path)
                    regrli[Model_List[i]] = {'model': model, 'type': 'rf'}
                    model_loaded = True
                except Exception as e:
                    print(f"Failed to load Random Forest model: {e}")
            
            if not model_loaded:
                raise FileNotFoundError(f"No model found for {variable_name}")
            
            # Load the corresponding scalar
            scalar_path = os.path.join(EmDir, Scaler_List[i] + ".joblib")
            try:
                scarlerli[Scaler_List[i]] = joblib.load(scalar_path)
            except Exception as e:
                print(f"Failed to load scalar: {e}")
                raise FileNotFoundError(f"No scalar found for {Scaler_List[i]}")
        
        return regrli, scarlerli
    return None, None


def CleanScaleObs(Obs_save, list1,model_type):
    """
    Clean and scale observation data
    
    Parameters:
    -----------
    Obs_save : DataFrame
        Raw observation data
    list1 : list
        List of observation variable names
        
    Returns:
    --------
    tuple : (Obs_save, obli) - cleaned observations and scalers dictionary
    """
    Obs_save['obs']=Obs_save['obs'].astype("float")
    if  model_type == "Xiulin_nn":
        print("nodate")
    else:
        Obs_save = Obs_save.dropna()
        Obs_save['Date'] = pd.to_datetime(Obs_save["Date"])
        Obs_save["year"] = Obs_save["Date"].dt.year
        Obs_save["month"] = Obs_save["Date"].dt.month
        Obs_save["DOY"] = Obs_save["Date"].dt.dayofyear


    # Dynamically create scalers and scale for all variables in list1
    obli = {}
    #print(Obs_save)
    for var in list1:
        # Handle possible zeros or negatives for log
        obs_vals = np.array(Obs_save.loc[Obs_save["Set"] == var, 'obs'])
        # Add small value to avoid log(0)
        obs_vals = np.log(np.abs(obs_vals) + 0.0001).reshape(-1, 1)
        scaler = StandardScaler().fit(obs_vals)
        Obs_save.loc[Obs_save["Set"] == var, 'obs'] = scaler.transform(obs_vals)
        obli[var] = scaler
    return Obs_save, obli


def TestLL(regrli, scarleri, initialpos, samples_sub, scaler_pars, log_prob_func):
    """
    Test log likelihood calculation at a specific position
    
    Parameters:
    -----------
    regrli : dict
        Dictionary of models
    scarleri : dict
        Dictionary of scalers
    initialpos : int
        Index of position to test
    samples_sub : DataFrame
        Subset of samples
    scaler_pars : StandardScaler
        Parameter scaler
    log_prob_func : callable
        Log probability function to test
        
    Returns:
    --------
    None : Prints results
    """
    initial_position = np.array(samples_sub.iloc[[initialpos]].T)
    print(initial_position)
    Logl = log_prob_func(regrli, scarleri, 
                         scaler_pars.transform(np.array(initial_position).flatten().reshape(1, -1)).flatten())
    print(Logl)


def AdaptiveMCMC(par_cov_matrix, steps, adapt_interval, burnin, dim, x_1, 
                 log_prob_func, regrli, scarleri, scaler_pars, samples_scale,
                 benchmark="./save.csv"):
    """
    Adaptive MCMC sampler
    
    Parameters:
    -----------
    par_cov_matrix : array
        Initial step matrix
    steps : int
        Number of steps to take in total
    adapt_interval : int
        How often to adapt the sampling range
    burnin : int
        How long before counting adjustments
    dim : int
        Dimensions in the parameter set
    x_1 : array
        Initial parameters
    log_prob_func : callable
        Log probability function
    regrli : dict
        Regression models
    scarleri : dict
        Scalers
    scaler_pars : StandardScaler
        Parameter scaler
    samples_scale : array
        Scaled samples for bounds
    benchmark : str
        Where to save the benchmarks
        
    Returns:
    --------
    DataFrame : movelog - all proposed moves
    """
    movelog = pd.DataFrame([])
    acceptlog = pd.DataFrame([])
    scorelog = np.array([])
    namelog = np.array([])
    acceptance_rates = []
    num_adapted_samples = 0
   
    x_1 = scaler_pars.transform(np.array(x_1).flatten().reshape(1, -1)).flatten()
    Logl = log_prob_func(regrli, scarleri, x_1)
    print("~~~~~~~~~~~~~~~~~~~Initial Logliklihood~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(Logl)
    
    sum_states = np.zeros(dim)
    outer_product_sum = np.zeros((dim, dim))
    num_adapted_samples = 0
    lag = 0
    plus = 0
    accepted_samples_for_adaptation = []
    Trueburn = burnin + adapt_interval
    
    for time in list(range(1, steps)):
        lag = lag + 1
        if time >= burnin and (time + 1) % adapt_interval == 0 and (len(accepted_samples_for_adaptation) > 1):
            # Update running sums for mean and covariance calculation
            par_cov_matrix = (np.cov(np.array(accepted_samples_for_adaptation).T)) * .2
            # Add a small regularization to prevent singular covariance matrix
            par_cov_matrix += np.eye(dim) * 1e-6
                   
        newproposed = np.random.multivariate_normal(np.array(x_1), par_cov_matrix, size=1).tolist()[0]
        newproposed = bounce(newproposed, pd.DataFrame(samples_scale))
        movelog = pd.concat([movelog, pd.DataFrame(newproposed)], axis=1)
        NewLogl = log_prob_func(regrli, scarleri, np.array(newproposed))
        NewLogl = NewLogl + plus
        
        if time % 200 == 0:
            print("~~~~Timestep~~~~~~~")
            print(time)
            print("Log-likelihood")
            print(Logl)
            
        scorelog = np.append(scorelog, Logl)
        
        if (acceptance(Logl, NewLogl)):
            Logl = NewLogl - plus
            plus = 0
            lag = 0
            x_1 = newproposed
            accepted_samples_for_adaptation.append(newproposed)
            
            if time > Trueburn:
                acceptlog = pd.concat([acceptlog, pd.DataFrame(newproposed)], axis=1)
                
        if (time % 500 == 0):
            pd.DataFrame(movelog).to_csv(benchmark)
            print('##########################################################################################')
            print("Total Accepted")
            print(len(acceptlog.T))
            
    print("Total Accepted")
    print(len(acceptlog.T))
    return movelog


# ============================================================================
# DREAM SAMPLER CLASS
# ============================================================================

class DREAMSampler:
    """
    DREAM (DiffeRential Evolution Adaptive Metropolis) Sampler
    Multi-chain MCMC method for efficient Bayesian parameter estimation
    """
    
    def __init__(self, 
                 log_posterior: Callable,
                 n_chains: int = 3,
                 n_params: int = 2,
                 delta: int = 3,
                 c: float = 0.1,
                 c_star: float = 1e-12,
                 n_crossover: int = 3,
                 p_gamma_unity: float = 0.2):
        """
        Initialize DREAM sampler
        
        Parameters:
        -----------
        log_posterior : Callable
            Function that returns log posterior probability
        n_chains : int
            Number of parallel chains (minimum 3, recommended: 3-10)
        n_params : int
            Number of parameters to estimate
        delta : int
            Number of chain pairs used for proposal generation
        c : float
            Scaling factor for epsilon randomization
        c_star : float
            Small constant to maintain ergodicity
        n_crossover : int
            Number of crossover probabilities to choose from
        p_gamma_unity : float
            Probability of setting gamma to 1
        """
        self.log_posterior = log_posterior
        self.n_chains = max(n_chains, 3)  # Minimum 3 chains
        self.n_params = n_params
        self.delta = min(delta, n_chains // 2)
        self.c = c
        self.c_star = c_star
        self.n_crossover = n_crossover
        self.p_gamma_unity = p_gamma_unity
        
        # Storage for chains
        self.chains = None
        self.log_posteriors = None
        self.acceptance_rate = []
        
    def initialize_chains(self, initial_positions: np.ndarray = None, 
                         bounds: List[Tuple[float, float]] = None):
        """
        Initialize chain positions
        
        Parameters:
        -----------
        initial_positions : np.ndarray, optional
            Initial positions for chains (n_chains x n_params)
        bounds : List[Tuple[float, float]], optional
            Parameter bounds for random initialization
        """
        if initial_positions is not None:
            if initial_positions.shape != (self.n_chains, self.n_params):
                raise ValueError(f"Initial positions must be shape ({self.n_chains}, {self.n_params})")
            self.chains = initial_positions.copy()
        else:
            # Random initialization within bounds
            if bounds is None:
                bounds = [(-5, 5)] * self.n_params
            
            self.chains = np.zeros((self.n_chains, self.n_params))
            for i in range(self.n_params):
                self.chains[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], self.n_chains)
        
        # Calculate initial log posteriors
        self.log_posteriors = np.array([self.log_posterior(chain) for chain in self.chains])
        
    def generate_crossover_probabilities(self) -> np.ndarray:
        """Generate crossover probability vector for parameter subspace sampling"""
        # Sample from predefined crossover values
        crossover_values = np.linspace(1/self.n_crossover, 1.0, self.n_crossover)
        CR = np.random.choice(crossover_values)
        
        # Create crossover mask
        z = np.random.uniform(0, 1, self.n_params)
        return (z <= CR).astype(float)
        
    def generate_proposal(self, chain_idx: int, all_chains: np.ndarray) -> np.ndarray:
        """
        Generate proposal using differential evolution
        
        Parameters:
        -----------
        chain_idx : int
            Index of current chain
        all_chains : np.ndarray
            Current positions of all chains
            
        Returns:
        --------
        proposal : np.ndarray
            Proposed new position
        """
        # Select delta pairs of chains (excluding current chain)
        available_chains = list(range(self.n_chains))
        available_chains.remove(chain_idx)
        
        # Sample 2*delta chains for delta pairs
        selected = np.random.choice(available_chains, size=2*self.delta, replace=False)
        chain_pairs = selected.reshape(self.delta, 2)
        
        # Calculate gamma (jump rate)
        if np.random.uniform() < self.p_gamma_unity:
            gamma = 1.0
        else:
            gamma = 2.38 / np.sqrt(2 * self.delta * self.n_params)
        
        # Generate crossover probabilities
        CR_mask = self.generate_crossover_probabilities()
        
        # Calculate differential evolution step
        de_step = np.zeros(self.n_params)
        for pair in chain_pairs:
            de_step += all_chains[pair[0]] - all_chains[pair[1]]
        
        # Add randomization
        epsilon = self.c * np.random.normal(0, 1, self.n_params)
        
        # Generate proposal with subspace sampling
        proposal = all_chains[chain_idx].copy()
        update_mask = CR_mask > 0
        
        if np.any(update_mask):
            proposal[update_mask] += (gamma * de_step[update_mask] + 
                                     epsilon[update_mask] + 
                                     self.c_star * np.random.normal(0, 1, np.sum(update_mask)))
        
        return proposal
    
    def metropolis_acceptance(self, current_log_post: float, 
                             proposal_log_post: float) -> bool:
        """Metropolis acceptance criterion"""
        if proposal_log_post > current_log_post:
            return True
        #print(proposal_log_post)
        #print(current_log_post)
        acceptance_prob = np.exp(proposal_log_post - current_log_post)
        return np.random.uniform() < acceptance_prob
    
    def detect_outliers(self, window: int = 50) -> np.ndarray:
        """
        Detect outlier chains using Interquartile Range (IQR) method
        
        Parameters:
        -----------
        window : int
            Number of recent iterations to consider
            
        Returns:
        --------
        outliers : np.ndarray
            Boolean array indicating outlier chains
        """
        if len(self.acceptance_rate) < window:
            return np.zeros(self.n_chains, dtype=bool)
        
        recent_accepts = np.array(self.acceptance_rate[-window:])
        chain_accept_rates = recent_accepts.mean(axis=0)
        
        Q1 = np.percentile(chain_accept_rates, 25)
        Q3 = np.percentile(chain_accept_rates, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (chain_accept_rates < lower_bound) | (chain_accept_rates > upper_bound)
        return outliers
    
    def run(self, n_iterations: int, burnin: int = 500, 
            thin: int = 1, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run DREAM sampling
        
        Parameters:
        -----------
        n_iterations : int
            Total number of iterations
        burnin : int
            Number of burnin iterations to discard
        thin : int
            Thinning interval for storing samples
        verbose : bool
            Print progress information
            
        Returns:
        --------
        samples : np.ndarray
            Posterior samples (n_samples x n_params)
        log_posts : np.ndarray
            Log posterior values for samples
        """
        if self.chains is None:
            raise ValueError("Chains not initialized. Call initialize_chains() first.")
        
        # Storage for all iterations
        all_chains_history = np.zeros((n_iterations, self.n_chains, self.n_params))
        all_log_posts_history = np.zeros((n_iterations, self.n_chains))
        
        # Progress tracking
        accepted_per_iteration = np.zeros((n_iterations, self.n_chains))
        
        for iteration in range(n_iterations):
            accepts = np.zeros(self.n_chains, dtype=bool)
            # Update each chain
            for chain_idx in range(self.n_chains):
                # Generate proposal
                proposal = self.generate_proposal(chain_idx, self.chains)
                
                # Calculate log posterior for proposal
                proposal_log_post = self.log_posterior(proposal)
                
                # Acceptance step
                if self.metropolis_acceptance(self.log_posteriors[chain_idx], 
                                              proposal_log_post):
                    self.chains[chain_idx] = proposal
                    self.log_posteriors[chain_idx] = proposal_log_post
                    accepts[chain_idx] = True
            
            # Store current state
            all_chains_history[iteration] = self.chains.copy()
            all_log_posts_history[iteration] = self.log_posteriors.copy()
            accepted_per_iteration[iteration] = accepts
            self.acceptance_rate.append(accepts)
            
            # Outlier detection and correction (after burnin)
            if iteration > burnin and iteration % 100 == 0:
                outliers = self.detect_outliers()
                if np.any(outliers):
                    # Replace outlier chains with random good chains
                    good_chains = np.where(~outliers)[0]
                    for outlier_idx in np.where(outliers)[0]:
                        replacement_idx = np.random.choice(good_chains)
                        self.chains[outlier_idx] = self.chains[replacement_idx] + \
                                                   np.random.normal(0, 0.01, self.n_params)
                        self.log_posteriors[outlier_idx] = self.log_posterior(self.chains[outlier_idx])
            
            # Progress reporting
            if (iteration + 1) % 100 == 0:
                accept_rate = np.mean(accepted_per_iteration[max(0, iteration-499):iteration+1])
                print(f"Iteration {iteration + 1}/{n_iterations} - "
                      f"Acceptance rate: {accept_rate:.3f} - "
                      f"Mean log-posterior: {np.mean(self.log_posteriors):.2f}")
        
        # Extract post-burnin samples with thinning
        samples = []
        log_posts = []
        
        for iteration in range(burnin, n_iterations, thin):
            for chain_idx in range(self.n_chains):
                samples.append(all_chains_history[iteration, chain_idx])
                log_posts.append(all_log_posts_history[iteration, chain_idx])
        
        samples = np.array(samples)
        log_posts = np.array(log_posts)
        
        
        final_accept_rate = np.mean(accepted_per_iteration)
        print(f"\nSampling complete!")
        print(f"Overall acceptance rate: {final_accept_rate:.3f}")
        print(f"Total samples collected: {len(samples)}")
        
        return samples, log_posts
    
    def get_chain_history(self) -> np.ndarray:
        """Return full chain history for diagnostics"""
        return np.array(self.acceptance_rate)
