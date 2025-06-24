from scipy import stats
from scipy.stats import lognorm
import re
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def nuts_sampler_multi(log_prob_func, grad_log_prob_func, initial_position, n_samples, step_size=0.1, max_tree_depth=10):
    """
    No-U-Turn Sampler (NUTS) for MCMC with multiple inputs
    
    Parameters:
    - log_prob_func: function to compute log probability of a position
    - grad_log_prob_func: function to compute gradient of log probability
    - initial_position: initial position (numpy array)
    - n_samples: number of samples to generate
    - step_size: initial step size for leapfrog integration
    - max_tree_depth: maximum depth of binary tree to explore
    
    Returns:
    - numpy array of samples
    """
    
    def leapfrog(position, momentum, step_size):
        """Perform one leapfrog step"""
        momentum_half = momentum + 0.5 * step_size * grad_log_prob_func(position)
        position_new = position + step_size * momentum_half
        momentum_new = momentum_half + 0.5 * step_size * grad_log_prob_func(position_new)
        return position_new, momentum_new
    
    def build_tree(position, momentum, u, v, j, step_size, log_u):
        """Recursively build tree for NUTS"""
        if j == 0:
            # Base case - take one leapfrog step in the direction v
            position_new, momentum_new = leapfrog(position, momentum, v * step_size)
            log_prob = log_prob_func(position_new)
            log_joint = log_prob - 0.5 * np.sum(momentum_new**2)
            n_accept = 1 if log_u <= log_joint else 0
            return position_new, momentum_new, position_new, momentum_new, position_new, n_accept, 1, log_joint > log_u - 1000
        else:
            # Recursion - build the left and right subtrees
            position_m, momentum_m, position_p, momentum_p, position_new, n_accept, n, valid = \
                build_tree(position, momentum, u, v, j - 1, step_size, log_u)
            if valid:
               
                if v == -1:
                    position_m, momentum_m, _, _, position_prime, n_accept_prime, n_prime, valid = \
                        build_tree(position_m, momentum_m, u, v, j - 1, step_size, log_u)
                else:
                    _, _, position_p, momentum_p, position_prime, n_accept_prime, n_prime, valid = \
                        build_tree(position_p, momentum_p, u, v, j - 1, step_size, log_u)
                #print(valid)
                if valid and np.random.rand() < n_prime / max(n + n_prime, 1):
                    #print(accept)
                    position_new = position_prime
                n_accept += n_accept_prime
                n += n_prime
                # Check if the stopping criterion is satisfied
                valid = valid and np.dot(position_p - position_m, momentum_m) >= 0 and \
                        np.dot(position_p - position_m, momentum_p) >= 0
            return position_m, momentum_m, position_p, momentum_p, position_new, n_accept, n, valid

    # Initialize
    position = initial_position
    n_dims = len(initial_position)
    samples = np.zeros((n_samples, n_dims))

    for i in range(n_samples):
        print(i)
        # Resample momentum
        momentum = np.random.randn(n_dims)
        
        # Initialize the binary tree
        position_m = position_p = position.copy()
        momentum_m = momentum_p = momentum.copy()
        
        # Initialize the slice variable
        log_u = log_prob_func(position) - 0.5 * np.sum(momentum**2) + np.log(np.random.rand())
        #print(log_u)
        # Initialize algorithm-specific variables
        j = 0
        n = 1
        s = 1
        
        while s == 1:
            # Choose a direction v uniformly at random
            v = 2 * (np.random.rand() < 0.5) - 1
            
            # Build a new subtree in the chosen direction
            if v == -1:
                position_m, momentum_m, _, _, position_prime, n_prime, n_double, s = \
                    build_tree(position_m, momentum_m, log_u, v, j, step_size, log_u)
            else:
                _, _, position_p, momentum_p, position_prime, n_prime, n_double, s = \
                    build_tree(position_p, momentum_p, log_u, v, j, step_size, log_u)
            
            # Use Metropolis-Hastings to decide whether to accept the new state
            #print(n_prime)
            #print(n)
            if s == 1 and np.random.rand() < min(1, n_prime / n):
                print('accept')
                position = position_prime.copy()
                print(position)
            
            n += n_prime
            j += 1
           # print(j)
            # Stop if we've gone too deep in the tree
            if j >= max_tree_depth:
                break
        
        samples[i] = position
        
    return samples


def boundby(single,min,max,name):
    if value < min_val:
        print("bounce")
        print(name)
        value = min_val + np.abs(min_val - value)
    elif value > max_val:
        print("bounce")
        print(name)
        value = max_val - (value - max_val)
    if value < min_val:
        print("bounce")
        print(name)
        value = min_val + np.abs(min_val - value)
    elif value > max_val:
        print("bounce")
        print(name)
        value = max_val - (value - max_val)
    # if the value still exceeds the range, reassign it randomly
    if value < min_val or value > max_val:
        value = np.random.uniform(min_val, max_val)
    return(single)

def bounce(new_proposed,samples):
       for i, col in enumerate(samples.columns):
        new_proposed.iloc[i, 0] = boundby(new_proposed.iloc[i, 0], samples[col].min(), samples[col].max(), name=col)
        return(newproposed)
# Example usage:
def log_prob(x):
    """
    Calculate the log probability of a given set of parameters.

    Parameters:
    x (array): The input parameters.

    Returns:
    float: The log probability of the input parameters.
    """
    # Inverse transform the input parameters
    x = list(scaler_pars.inverse_transform(x.reshape(1, -1)))
    x = list(x)

    x = pd.DataFrame({
        'p_taper': x[0][0],
        'ball_berry_slope': x[0][1],
        'Vcmax25': x[0][2],
        'sla_top': x[0][3],
        'p50_node_aroot': x[0][4],
        'rs2': x[0][5]
    }, index=[0])
    # Initialize the output
    #print(x)
    DFout=0
    #DFout=pd.DataFrame({})
    # loop through all our model types
    for i in list(range(0,len(Model_List))):
        #print(i)
        # load new model/scaler
        regr = regrli[Model_List[i]]
        scaler = scarlerli[Scaler_List[i]]
        Weight = Weight_list[i]
        ob_set=Obs_save[Obs_save["Set"]==Obs_List[i]]
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
            one = one.set_axis(Names, axis=1)

        one= scaler.transform(one)
        predicty=regr.predict(one)
        Frame=pd.concat([ob_set.reset_index(drop=True),pd.Series(predicty,name="sim").reset_index(drop=True)],axis=1)
        Frame.loc[Frame['sim']<.00001,0]=0.0001
        if (Obs_List[i] =="LWPmin")|(Obs_List[i] =="LWPmax"):
            Frame["obs"]=abs(Frame["obs"])
            Frame["sim"]=abs(Frame["sim"])
        #DFout=pd.concat([DFout,Frame])
        #print(len(DFout))
        ll1 = np.sum(lognorm.pdf(Frame['sim'], s=Frame['error'], scale=np.exp(Frame["obs"])))

        # Update the outputg
        DFout += ll1 * Weight

        # Check physical possibility
        p =sum([1 for t in range(len(x.columns)) if (x.iloc[0, t] < samples[[x.columns[t]]].min().values) or (x.iloc[0, t] > samples[[x.columns[t]]].max().values)])
        if p >=1:
            DFout= -np.inf
    DFout=(DFout)
    return (DFout)
    
def grad_log_prob(x):
    return -x
    
    
samples=pd.read_csv('C:/Users/345578/Desktop/ML_scaler/LHS.sam.csv')
Varset= ['p_taper',
       'ball_berry_slope', 'Vcmax25', 'sla_top', 'p50_node_aroot','rs2']
samples_sub=samples[Varset]
scaler_pars =StandardScaler().fit(samples_sub.values)

Names=pd.concat([pd.Series(samples_sub.columns),pd.Series(["year","month",])])
Names2=pd.concat([pd.Series(samples_sub.columns),pd.Series(["Year","DOY",])])


LOADIN=True
Emdir = "C:/Users/345578/Desktop/NewML/ML_models/"
Obsdir="C:/Users/383517/Desktop/LT2024/Machine_Learning/FATES_emulator/"
Datadir="C:/Users/383517/Desktop/LT2024/Machine_Learning/Data/"

Model_List = ["ET_full_012825", "GPP_minvar_012825",'H2oSOI1_minvar_012825',"LWP_max_model_012825",
             "LWP_min_model_012825","RO_minvar_031125"]

Scaler_List = ["ET_full_012825_Scalar","GPP_minvar_031125_Scalar","H2oSOI1_minvar_012825_Scalar",
              "LWP_max_model_012825_Scalar","LWP_min_model_012825_Scalar","RO_minvar_031125_Scalar"]


Weight_list=[1.0,1.0,1,1.0,0.02,.02,1]

Obs_List = ["ET",'MonthlyGPP', 'SWC10',"LWPmax","LWPmin",'Runoff']
if LOADIN== True: 
    regrli={}
    scarlerli={}
    for i in list(range(0,len(Model_List))) :
        print(i)
    
     # load new model/scaler
        variable_name=Model_List[i]
        regrli[Model_List[i]]=joblib.load(Emdir+Model_List[i]+'.joblib')
        print(regrli)
        scarlerli[Scaler_List[i]]=joblib.load(Emdir+Scaler_List[i]+'.joblib')

list1=['MonthlyGPP', 'Runoff', 'SWC10', 'SWC40', 'SWC100', 'ET', 'LWPmin',
       'LWPmax']
       
       
Obs_save=pd.read_csv("C:/Users/345578/Desktop/NewML/Synth/Syntheticg1_31125.csv").iloc[:,1:]
Obs_save=Obs_save.dropna()
Obs_save['Date']=pd.to_datetime(Obs_save["Date"])
Obs_save["year"]=Obs_save["Date"].dt.year
Obs_save["month"]=Obs_save["Date"].dt.month
Obs_save["DOY"]=Obs_save["Date"].dt.dayofyear

Obs_save.loc[Obs_save["Set"]==list1[0],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[0],'obs'].mean()*0.001)
Obs_save.loc[Obs_save["Set"]==list1[1],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[1],'obs'].mean()*.001)
Obs_save.loc[Obs_save["Set"]==list1[2],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[2],'obs'].mean()*.001)
Obs_save.loc[Obs_save["Set"]==list1[3],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[3],'obs'].mean()*.001)
Obs_save.loc[Obs_save["Set"]==list1[4],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[4],'obs'].mean()*.001)
Obs_save.loc[Obs_save["Set"]==list1[5],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[5],'obs'].mean()*.001)
Obs_save.loc[Obs_save["Set"]==list1[6],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[6],'obs'].mean()*.0001)
Obs_save.loc[Obs_save["Set"]==list1[7],'error']=abs(Obs_save.loc[Obs_save["Set"]==list1[7],'obs'].mean()*.0001)


initialpos=11

#initial_position=#[samples_sub.iloc[[initialpos]].T,samples_sub]
initial_position=np.array(samples_sub.iloc[[initialpos]].T)

Logl=log_prob(scaler_pars.transform(initial_position.reshape(1, -1)))

Logl
samples_nuts = nuts_sampler_multi(log_prob, grad_log_prob,  scaler_pars.transform(initial_position.reshape(1, -1))[0], max_tree_depth=5, n_samples=100,step_size=.1)

a=scaler_pars.inverse_transform(samples_nuts )
b=pd.DataFrame(a)

RUNname
b.to_csv("C:/Users/345578/Desktop/NewML/MCMC_runs/"+RUNname+".csv" )