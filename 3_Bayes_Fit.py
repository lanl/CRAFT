"""
CRAFT Bayesian Parameter Fitting Script
Performs MCMC sampling using either Adaptive MCMC or DREAM algorithm
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config_utils import get_bayes_settings, get_emulator_metadata
from bayes_functions import (
    log_prob, Loadmodels, CleanScaleObs, TestLL, AdaptiveMCMC, DREAMSampler
)

# Set working directory to script location
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
os.chdir(script_directory)

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

print("="*70)
print("CRAFT Bayesian Parameter Fitting")
print("="*70)

# Load settings from XML config file
Settings = get_bayes_settings()

initialpos = int(Settings.loc['initialpos', 'value'])
steps = int(Settings.loc['steps', 'value'])
adapt_interval = int(Settings.loc['adapt_interval', 'value'])
burnin = int(Settings.loc['burnin', 'value'])
Model_List = eval(Settings.loc['Model_List', 'value'])
Scaler_List = eval(Settings.loc['Scaler_List', 'value'])
Obs_List = eval(Settings.loc['Obs_List', 'value'])
list1 = eval(Settings.loc['list1', 'value'])
samples_file = Settings.loc['samples_file', 'value']
obsfile = Settings.loc['obsfile', 'value']
output_file = Settings.loc['output_file', 'value']

print(f"\nSettings loaded from XML configuration")
print(f"  Samples file: {samples_file}")
print(f"  Observations file: {obsfile}")
print(f"  Output file: {output_file}")

# Load model configuration to determine model types
try:
    Meta = get_emulator_metadata()
    model_type = "rf"  # default
    if "model_type" in Meta["Var"].values:
        model_type = Meta.loc[Meta["Var"]=='model_type']['Path'].values[0]
    print(f"  Model type: {model_type}")
except:
    model_type = "rf"
    print(f"  Model type: {model_type} (default)")

# ============================================================================
# LOAD DATA
# ============================================================================

samples = pd.read_csv(samples_file)

Emdir = "data/Models/"
if model_type == "rf":
    Varset = pd.read_csv("diag/FitOrder.csv")
else: 
    s_Varset = pd.Series(samples.columns.tolist())
    s_Varset = pd.concat([s_Varset, pd.Series(["month", "year"])])
    print(s_Varset)
    ncols = 2 + len(samples.columns.tolist())
    Varset = pd.DataFrame({"r": list(range(0, len(s_Varset))), "0": s_Varset})
    print(Varset)

# Extract variable names
Varset = Varset.iloc[:, 1].values

# Remove time variables
indices_to_delete = np.where(Varset == "month")[0]
Varset2 = np.delete(Varset, indices_to_delete)
indices_to_delete = np.where(Varset2 == "year")[0]
Varset2 = np.delete(Varset2, indices_to_delete)
indices_to_delete = np.where(Varset2 == "DOY")[0]
Varset2 = np.delete(Varset2, indices_to_delete)
print(f"\nParameters to estimate: {Varset2}")

# Prepare parameter scaling
samples_sub = samples[Varset2]
scaler_pars = StandardScaler().fit(samples_sub.values)
samples_scale = scaler_pars.transform(samples_sub)
Names = pd.concat([pd.Series(samples_sub.columns), pd.Series(["year", "month"])])
Names2 = pd.concat([pd.Series(samples_sub.columns), pd.Series(["year", "DOY"])])

# Load and prepare observations
Obs_save = pd.read_csv(obsfile).iloc[:, 1:]
print(f"\nObservation data loaded:")
print(Obs_save.head())
Obs_save = Obs_save.dropna()
Obs_save, obli = CleanScaleObs(Obs_save, list1)

dims = len(Varset2)
S = np.repeat(1, dims)

# ============================================================================
# LOAD MODELS
# ============================================================================

print("\n" + "="*70)
print("Loading Emulator Models")
print("="*70)

LOADIN = True
if LOADIN:
    regrli, scarleri = Loadmodels(LOADIN, Emdir, Model_List, Scaler_List, model_type)

# ============================================================================
# DEFINE LOG PROBABILITY WRAPPER
# ============================================================================

def log_prob_wrapper(regrli_local, scarleri_local, x):
    """Wrapper to provide all necessary arguments to log_prob"""
    return log_prob(
        regrli_local, scarleri_local, x, scaler_pars, Varset2, samples, 
        Model_List, Scaler_List, Obs_List, Obs_save, Varset, Names, Names2, obli
    )

# ============================================================================
# TEST LOG LIKELIHOOD
# ============================================================================

print("\n" + "="*70)
print("Testing Log Likelihood at Initial Position")
print("="*70)

TestLL(regrli, scarleri, initialpos=111, samples_sub=samples_sub, 
       scaler_pars=scaler_pars, log_prob_func=log_prob_wrapper)

# ============================================================================
# RUN MCMC SAMPLING
# ============================================================================

sampler_method = Settings.loc['sampler_method', 'value'].lower()
print(f"\n{'='*70}")
print(f"MCMC Sampler: {sampler_method.upper()}")
print(f"{'='*70}\n")

if sampler_method == 'dream':
    # ========================================================================
    # DREAM SAMPLER
    # ========================================================================
    # Load DREAM parameters
    dream_n_chains = int(Settings.loc['dream_n_chains', 'value'])
    dream_delta = int(Settings.loc['dream_delta', 'value'])
    dream_c = float(Settings.loc['dream_c', 'value'])
    dream_c_star = float(Settings.loc['dream_c_star', 'value'])
    dream_n_crossover = int(Settings.loc['dream_n_crossover', 'value'])
    dream_p_gamma_unity = float(Settings.loc['dream_p_gamma_unity', 'value'])
    dream_thin = int(Settings.loc['dream_thin', 'value'])
    
    print(f"DREAM Configuration:")
    print(f"  Number of chains: {dream_n_chains}")
    print(f"  Delta (chain pairs): {dream_delta}")
    print(f"  Total iterations: {steps}")
    print(f"  Burnin: {burnin}")
    print(f"  Thinning: {dream_thin}")
    print(f"\n~~~~~~~~~~Beginning DREAM sampling~~~~~~~~~~~~~~~~~~~~~~~")
    
    # Define log posterior wrapper for DREAM
    def log_posterior_dream(params):
        """Wrapper for log_prob to work with DREAM sampler"""
        x_scaled = scaler_pars.transform(params.reshape(1, -1)).flatten()
        return log_prob_wrapper(regrli, scarleri, x_scaled)
    
    # Initialize DREAM sampler
    sampler = DREAMSampler(
        log_posterior=log_posterior_dream,
        n_chains=dream_n_chains,
        n_params=dims,
        delta=dream_delta,
        c=dream_c,
        c_star=dream_c_star,
        n_crossover=dream_n_crossover,
        p_gamma_unity=dream_p_gamma_unity
    )
    
    # Initialize chains from samples
    if dream_n_chains <= len(samples_sub):
        initial_positions = samples_sub.sample(dream_n_chains, random_state=42).values
    else:
        # If need more chains than samples, duplicate with noise
        base_samples = samples_sub.sample(dream_n_chains, replace=True, random_state=42).values
        noise = np.random.randn(dream_n_chains, dims) * 0.01
        initial_positions = base_samples + noise
    
    sampler.initialize_chains(initial_positions=initial_positions)
    
    # Run DREAM
    samples_dream, log_posts_dream = sampler.run(
        n_iterations=steps,
        burnin=burnin,
        thin=dream_thin,
        verbose=True
    )
    
    # Convert to DataFrame and save
    Fixedlog = pd.DataFrame(samples_dream, columns=Varset2)
    Fixedlog['log_posterior'] = log_posts_dream
    Fixedlog.to_csv(output_file, index=False)
    
    print(f"\nDREAM sampling completed!")
    print(f"Total posterior samples: {len(Fixedlog)}")
    print(f"Results saved to: {output_file}")
    
elif sampler_method == 'adaptive':
    # ========================================================================
    # ADAPTIVE MCMC
    # ========================================================================
    print(f"Adaptive MCMC Configuration:")
    print(f"  Total iterations: {steps}")
    print(f"  Adaptation interval: {adapt_interval}")
    print(f"  Burnin: {burnin}")
    print(f"\n~~~~~~~~~~Beginning Adaptive MCMC sampling~~~~~~~~~~~~~~~~~~~~~~~")
    
    par_cov_matrix = np.outer(S, S)
    log = AdaptiveMCMC(
        par_cov_matrix,
        steps,
        adapt_interval,
        burnin,
        dims,
        samples_sub.iloc[[initialpos]].T,
        log_prob_wrapper,
        regrli,
        scarleri,
        scaler_pars,
        samples_scale,
        benchmark="bm.csv"
    )
    
    Fixedlog = pd.DataFrame(scaler_pars.inverse_transform(log.T))
    Fixedlog.to_csv(output_file)
    
    print(f"\nAdaptive MCMC sampling completed!")
    print(f"Results saved to: {output_file}")
    
else:
    raise ValueError(f"Unknown sampler method: {sampler_method}. Choose 'adaptive' or 'dream'.")

print(f"\n{'='*70}")
print("MCMC Sampling Complete!")
print(f"{'='*70}")
