import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
import re
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.stats import lognorm
from scipy.stats import norm
import os
import tensorflow as tf
from config_utils import get_bayes_settings, get_emulator_metadata

# Import DREAM sampler
try:
    from DREAM_example import DREAMSampler
    DREAM_AVAILABLE = True
except ImportError:
    DREAM_AVAILABLE = False
    print("Warning: DREAM_example.py not found. DREAM sampler will not be available.")

script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
# Change the current working directory to the script's directory
os.chdir(script_directory)

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

# Load model configuration to determine model types
try:
    Meta = get_emulator_metadata()
    # Check if model_type is specified in metadata
    model_type = "rf"  # default
    if "model_type" in Meta["Var"].values:
        model_type = Meta.loc[Meta["Var"]=='model_type']['Path'].values[0]
    print(f"Using model type: {model_type}")
except:
    model_type = "rf"  # Fall back to RF if metadata not found
    print(f"Metadata file not found or doesn't specify model type. Defaulting to Random Forest.")

samples = pd.read_csv(samples_file)
print(f"Settings loaded from XML configuration")

Emdir = "data/Models/"
if  model_type == "rf":
    Varset = pd.read_csv("diag/FitOrder.csv")
else: 
    s_Varset=pd.Series(samples.columns.tolist())
    s_Varset=pd.concat([s_Varset,pd.Series(["month","year"])])
    print(s_Varset)
    ncols=2+len(samples.columns.tolist())
 
    Varset=pd.DataFrame({"r":list(range(0,len(s_Varset))),"0":s_Varset})
    print(Varset)

def log_prob(regrli, scarlerli, x):
    """
    Calculate the log probability of a given set of parameters.

    Parameters:
    x (array): The input parameters. Transformed. 

    Returns:
    float: The log probability of the input parameters.
    """
    # Inverse transform the input parameters
    x = scaler_pars.inverse_transform(x.reshape(1, -1))
   # 
    outs = pd.DataFrame({}, index=[0])
    for i in list(range(0,len(Varset2))):
        outs[Varset2[i]]=x[0][i]
    x=outs
    #    print(x)

    # Initialize the output
    DFout=pd.DataFrame({})
    # loop through all our model types
    for i in list(range(0,len(Model_List))):
        # load new model/scaler
        model_info = regrli[Model_List[i]]
        regr = model_info['model']
        model_type = model_info['type']
        scaler = scarlerli[Scaler_List[i]]
        obscaler = obli[Obs_List[i]]
        
        ob_set = Obs_save[Obs_save["Set"] == Obs_List[i]]
        # Dynamically check for column
        if (Obs_List[i] =="LWPmin")|(Obs_List[i] =="LWPmax"):
            my = ob_set[["year", "DOY"]].reset_index(drop=True)
            my["Year"] = my["year"]
            my = my[["Year", "DOY"]]
            ob_set = ob_set.sort_values(["year", "DOY"])
            x_1_run_set = pd.concat([pd.concat([x] * len(my)).reset_index(), my], axis=1)
            one = x_1_run_set.iloc[:, 1:].sort_values(["Year", "DOY"])
            one = one.set_axis(Names2, axis=1)
        else: 
            my = ob_set[["year", "month"]].reset_index(drop=True)
            ob_set = ob_set.sort_values(["year", "month"])
            x_1_run_set = pd.concat([pd.concat([x] * len(my)).reset_index(), my], axis=1)
            one = x_1_run_set.iloc[:, 1:].sort_values(["year", "month"])
            one=one[Varset]
            #print(one)
           # one = one.set_axis(Names, axis=1)  
        #print(one)          
        one = scaler.transform(one)
        
        # Make predictions based on model type
        if model_type == "rf":
            predicty = regr.predict(one)
        else:  # Neural network
            # Convert to tensor for TF models
            one_tensor = tf.convert_to_tensor(one.astype('float32'))
            predicty = regr.predict(one_tensor, verbose=0).flatten()
            predicty[predicty < 0] = 0
          #  print(predicty )
        if (Obs_List[i] =="LWPmin")|(Obs_List[i] =="LWPmax"):
            predicty=np.abs(predicty)
        #print(predicty)    
        predicty=obscaler.transform(np.log(predicty+0.0001).reshape(-1, 1))
      
        Frame=pd.concat([ob_set.reset_index(drop=True),pd.Series(predicty.flatten(),name="sim").reset_index(drop=True)],axis=1)
        if (Obs_List[i] =="LWPmin")|(Obs_List[i] =="LWPmax"):
            Frame=Frame.sample(n=169,random_state=55)                    
        #print(Frame)
     #### This is a dataframe with the obs/sim/ in it
        DFout=pd.concat([DFout,Frame])

    #print(DFout)
    #addsim=DFout.loc[DFout["Set"]=='MonthlyGPP']["sim"]/DFout.loc[DFout["Set"]=='ET']["sim"]
    #addobs=DFout.loc[DFout["Set"]=='MonthlyGPP']["obs"]/DFout.loc[DFout["Set"]=='ET']["obs"]
    #### Calculate full ll
    #print(DFout['sim'],DFout['obs'])
    ll1=np.sum(norm.logpdf(DFout['sim'], loc=DFout['obs'], scale=20.0))
    #ll2=np.sum(norm.logpdf(addsim, loc=addobs, scale=5.0))
    #ll1+=ll2
    # Check physical possibility/Values are within the bounds of initial sample. 
    p =sum([1 for t in range(len(x.columns)) if (x.iloc[0, t] < samples[[x.columns[t]]].min().values) or (x.iloc[0, t] > samples[[x.columns[t]]].max().values)])
    if p >=1:
        ll1= -np.inf
    return (ll1)

def boundby(single,min,max):
    if single<min :
        single=min+np.abs(min-single)
    if single>max:
        single=max-(single-max)
    if single<min :
        single=min+np.abs(min-single)
    if single>max:
        single=max-(single-max)
    if single>max:
       single=np.random.uniform(min,max)
    if single<min:
       single=np.random.uniform(min,max)
    return(single)

def bounce(newproposed,samples_scale):
    for i in list(range(0,len(samples_scale.columns))):### Maybe this isn't minus 1
        newproposed[i]=boundby(newproposed[i],samples_scale.iloc[:,i].min(),samples_scale.iloc[:,i].max())
    return(newproposed)


def acceptance(x_likelihood, x_new_likelihood):
  if x_new_likelihood>x_likelihood:
        #rint("True_accept")
        return True
  else:
        accept=np.random.uniform(0,1)
        if accept < np.exp(x_new_likelihood-x_likelihood):
         #   print("roll accept")
            return True
        else:
            return False
        #return(np.exp(x_likelihood-x_new_likelihood))

def Loadmodels(LOADIN, EmDir, Model_List, Scaler_List):
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
            if model_type=="nn":
                try:
                    print(f"Loading neural network model: {nn_model_path}")
                    model = tf.keras.models.load_model(nn_model_path)
                    regrli[Model_List[i]] = {'model': model, 'type': 'nn'}
                    model_loaded = True
                except Exception as e:
                    print(f"Failed to load neural network model: {e}")
            
            # Try to load Random Forest model if NN didn't work or doesn't exist
            if model_type=="rf":
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

def CleanScaleObs(Obs_save):
    Obs_save=Obs_save.dropna()
    Obs_save['Date']=pd.to_datetime(Obs_save["Date"])
    Obs_save["year"]=Obs_save["Date"].dt.year
    Obs_save["month"]=Obs_save["Date"].dt.month
    Obs_save["DOY"]=Obs_save["Date"].dt.dayofyear

    #### Dynamically create scalers and scale for all variables in list1
    obli = {}
    for var in list1:
        # Handle possible zeros or negatives for log
        obs_vals = np.array(Obs_save.loc[Obs_save["Set"]==var, 'obs'])
        # Add small value to avoid log(0)
        obs_vals = np.log(np.abs(obs_vals) + 0.0001).reshape(-1, 1)
        scaler = StandardScaler().fit(obs_vals)
        Obs_save.loc[Obs_save["Set"]==var, 'obs'] = scaler.transform(obs_vals)
        obli[var] = scaler
    return(Obs_save, obli)

def TestLL(regrli,scarleri,initialpos=225):
    #initial_positin=#[samples_sub.iloc[[initialpos]].T,samples_sub]
    initial_position=np.array(samples_sub.iloc[[initialpos]].T)
    print(initial_position)
    Logl=log_prob(regrli,scarleri,scaler_pars.transform(np.array(initial_position).flatten().reshape(1, -1)).flatten())
    print(Logl)

def AdaptiveMCMC(par_cov_matrix,steps,adapt_interval,burnin,dim,x_1,benchmark="./save.csv"):
    '''
    par_cov_matrix: initial step matrix
    steps: number of steps to take in total
    adapt_interval: How often to adapt the sampling range
    burnin: How long before counting adjustments
    dim: Dimensions in the parameter set
    x_1: inital parameters in this script ex samples_sub.iloc[[250]].T
    benchmark: where to save the benchmarks for saving. 

    returns movelog, acceptlog, scorelog
    '''
    
    movelog=pd.DataFrame([])
    acceptlog=pd.DataFrame([])
    scorelog=np.array([])
    namelog=np.array([])
    acceptance_rates = []
    num_adapted_samples=0
   
    x_1=scaler_pars.transform(np.array(x_1).flatten().reshape(1, -1)).flatten()
    Logl=log_prob(regrli,scarleri,x_1)
    print("~~~~~~~~~~~~~~~~~~~Initial Logliklihood~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(Logl)
    sum_states = np.zeros(dim)
    outer_product_sum = np.zeros((dim, dim))
    num_adapted_samples = 0
    lag=0
    plus=0
    accepted_samples_for_adaptation=[]
    Trueburn=burnin+adapt_interval
    for time in list(range(1,steps)):
        #print(time)
        lag=lag+1
        if time >= burnin and (time + 1) % adapt_interval == 0 and (len(accepted_samples_for_adaptation) > 1):
                # Update running sums for mean and covariance calculation
                   #print(accepted_samples_for_adaptation)
                    par_cov_matrix= (np.cov(np.array(accepted_samples_for_adaptation).T))*.2
                    ####print(par_cov_matrix)
                    # Add a small regularization to prevent singular covariance matrix
                    par_cov_matrix += np.eye(dim) * 1e-6
                   
    
        newproposed=np.random.multivariate_normal(np.array(x_1),par_cov_matrix, size=1).tolist()[0]
        newproposed=bounce(newproposed,pd.DataFrame(samples_scale))
        movelog=pd.concat([movelog,pd.DataFrame(newproposed)],axis=1)
        NewLogl=log_prob(regrli,scarleri,np.array(newproposed))
        NewLogl=NewLogl+plus
        if time % 200 == 0:
            print("~~~~Timestep~~~~~~~")
            print(time)
            print("Log-likelihood")
            print(Logl)
        #print(NewLogl)  
        scorelog=np.append(scorelog,Logl)
        if (acceptance(Logl,NewLogl)):
            Logl=NewLogl-plus
            plus=0
            lag=0
            
            x_1=newproposed
            accepted_samples_for_adaptation.append(newproposed)
            #print('Accept')
            #print(len(accepted_samples_for_adaptation))
            if time >Trueburn:
                acceptlog=pd.concat([acceptlog,pd.DataFrame(newproposed)],axis=1)
                
        if (time % 500 == 0 ):
            pd.DataFrame(movelog).to_csv(benchmark)
            print('##########################################################################################')
            print("Total Accepted")
            print(len(acceptlog.T))
    print("Total Accepted")
    print(len(acceptlog.T))
    return(movelog)

Varset=Varset.iloc[:,1].values

indices_to_delete = np.where(Varset == "month")[0]
Varset2 = np.delete(Varset, indices_to_delete)
indices_to_delete = np.where(Varset2 == "year")[0]
Varset2 = np.delete(Varset2, indices_to_delete)
indices_to_delete = np.where(Varset2 =="DOY")[0]
Varset2 = np.delete(Varset2, indices_to_delete)
print(Varset2)
samples_sub=samples[Varset2]
scaler_pars =StandardScaler().fit(samples_sub.values)
samples_scale=scaler_pars.transform(samples_sub)
Names=pd.concat([pd.Series(samples_sub.columns),pd.Series(["year","month",])])
Names2=pd.concat([pd.Series(samples_sub.columns),pd.Series(["Year","DOY",])])

Obs_save=pd.read_csv(obsfile).iloc[:,1:]
print(Obs_save)
Obs_save=Obs_save.dropna()
Obs_save,obli=CleanScaleObs(Obs_save)
dims=len(Varset2)
S=np.repeat(1,dims)
#S=[1,1,1,1,1,1,1,1,1,10]

LOADIN=True
if LOADIN==True:
    regrli,scarleri=Loadmodels(LOADIN,Emdir,Model_List,Scaler_List)

TestLL(regrli,scarleri,initialpos=111)

# Load sampler configuration
sampler_method = Settings.loc['sampler_method', 'value'].lower()
print(f"\n{'='*70}")
print(f"MCMC Sampler: {sampler_method.upper()}")
print(f"{'='*70}\n")

# Run MCMC sampling based on selected method
if sampler_method == 'dream':
    # DREAM Sampler
    if not DREAM_AVAILABLE:
        raise ImportError("DREAM sampler selected but DREAM_example.py not found!")
    
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
        # params is already in original scale from DREAM
        x_scaled = scaler_pars.transform(params.reshape(1, -1)).flatten()
        return log_prob(regrli, scarleri, x_scaled)
    
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
    # Original Adaptive MCMC
    print(f"Adaptive MCMC Configuration:")
    print(f"  Total iterations: {steps}")
    print(f"  Adaptation interval: {adapt_interval}")
    print(f"  Burnin: {burnin}")
    print(f"\n~~~~~~~~~~Beginning Adaptive MCMC sampling~~~~~~~~~~~~~~~~~~~~~~~")
    
    par_cov_matrix = np.outer(S, S)
    log = AdaptiveMCMC(par_cov_matrix,
                 steps,
                 adapt_interval,
                 burnin, dims,
                 samples_sub.iloc[[initialpos]].T,
                 benchmark="bm.csv")
    
    Fixedlog = pd.DataFrame(scaler_pars.inverse_transform(log.T))
    Fixedlog.to_csv(output_file)
    
    print(f"\nAdaptive MCMC sampling completed!")
    print(f"Results saved to: {output_file}")
    
else:
    raise ValueError(f"Unknown sampler method: {sampler_method}. Choose 'adaptive' or 'dream'.")

print(f"\n{'='*70}")
print("MCMC Sampling Complete!")
print(f"{'='*70}")
