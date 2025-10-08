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

script_path = os.path.abspath(__file__)
# Get the directory containing the script
script_directory = os.path.dirname(script_path)
# Change the current working directory to the script's directory
os.chdir(script_directory)

initialpos=45
steps=50000
adapt_interval=1000
burnin=100
Model_List = ["CRAFT_practice_FullGPP","CRAFT_practice_FullH201"]
Scaler_List = ["CRAFT_practice_FullGPP_Scalar","CRAFT_practice_FullH201_Scalar"]
Obs_List = ['MonthlyGPP', 'SWC10']
list1=['MonthlyGPP','SWC10']
samples=pd.read_csv('Example_Obs_data/LHS.sam.csv')
obsfile="Example_Obs_data/Syntheticg_Set2_63025.csv"




Emdir = "data/Models/"



Varset=pd.read_csv("diag/FitOrder.csv")
def log_prob(regrli,scarlerli,x):
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
        regr = regrli[Model_List[i]]
        scaler = scarlerli[Scaler_List[i]]
        obscaler = obli[Obs_List[i]]
        
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
            
            one=one[Varset]
            #print(one)
           # one = one.set_axis(Names, axis=1)
            
        one= scaler.transform(one)
        predicty=regr.predict(one)
        #print(one)
        
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
    ll1=np.sum(norm.logpdf(DFout['sim'], loc=DFout["obs"], scale=5.0))
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
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        #print(np.exp(x_new_likelihood-x_likelihood))
        if accept < np.exp(x_new_likelihood-x_likelihood):
         #   print("roll accept")
            return True
        else:
            return False
        #return(np.exp(x_likelihood-x_new_likelihood))

def Loadmodels(LOADIN,EmDir,Model_List,Scaler_List):
    if LOADIN== True: 
        regrli={}
        scarlerli={}
        for i in list(range(0,len(Model_List))) :
        #   print(i)
         # load new model/scaler
            variable_name=Model_List[i]
            regrli[Model_List[i]]=joblib.load(Emdir+Model_List[i]+'.joblib')
            print(regrli)
            scarlerli[Scaler_List[i]]=joblib.load(Emdir+Scaler_List[i]+'.joblib')
        return(regrli,scarlerli) 
#regrli,scarleri=Loadmodels(LOADIN=True,
#"C:/Users/345578/Documents/Github/CRAFT/data/Models/",
 #       ["ET_full_012825", "GPP_minvar_012825",'H2oSOI1_minvar_012825',"LWP_max_model_012825",
 #            "LWP_min_model_012825","RO_minvar_031125"],["ET_full_012825_Scalar","GPP_minvar_031125_Scalar","H2oSOI1_minvar_012825_Scalar",
 #             "LWP_max_model_012825_Scalar","LWP_min_model_012825_Scalar","RO_minvar_031125_Scalar"]

def CleanScaleObs(Obs_save):
    Obs_save=Obs_save.dropna()
    Obs_save['Date']=pd.to_datetime(Obs_save["Date"])
    Obs_save["year"]=Obs_save["Date"].dt.year
    Obs_save["month"]=Obs_save["Date"].dt.month
    Obs_save["DOY"]=Obs_save["Date"].dt.dayofyear

    #### Create scaler and scale and scale
    GPP_obsscale=StandardScaler().fit(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[0],'obs']))).reshape(-1, 1))
    #ET_obsscale=StandardScaler().fit(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[1],'obs']))).reshape(-1, 1))
    SWC_obsscale=StandardScaler().fit(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[1],'obs']))).reshape(-1, 1))
    #SWC_obsscale=StandardScaler().fit(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[2],'obs']))).reshape(-1, 1))
    #Min_obsscale=StandardScaler().fit(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[6],'obs']+0.0001))).reshape(-1, 1))
    #Max_obsscale=StandardScaler().fit(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[7],'obs']+0.0001))).reshape(-1, 1))
    Obs_save.loc[Obs_save["Set"]==list1[0],'obs']=GPP_obsscale.transform(np.log(np.array(Obs_save.loc[Obs_save["Set"]==list1[0],'obs']).reshape(-1, 1)))
    #Obs_save.loc[Obs_save["Set"]==list1[1],'obs']=ET_obsscale.transform(np.log(np.array(Obs_save.loc[Obs_save["Set"]==list1[1],'obs']).reshape(-1, 1)))
    Obs_save.loc[Obs_save["Set"]==list1[1],'obs']=SWC_obsscale.transform(np.log(np.array(Obs_save.loc[Obs_save["Set"]==list1[1],'obs']).reshape(-1, 1)))
    #Obs_save.loc[Obs_save["Set"]==list1[5],'obs']=ET_obsscale.transform(np.log(np.array(Obs_save.loc[Obs_save["Set"]==list1[5],'obs']).reshape(-1, 1)))
    #Obs_save.loc[Obs_save["Set"]==list1[6],'obs']=Min_obsscale.transform(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[6],'obs']+0.0001))).reshape(-1, 1))
    #Obs_save.loc[Obs_save["Set"]==list1[7],'obs']=Max_obsscale.transform(np.log(np.abs(np.array(Obs_save.loc[Obs_save["Set"]==list1[7],'obs']+0.0001))).reshape(-1, 1))
    
    Obi = ['MonthlyGPP', 'SWC10',]
    obli={}
    obli[Obi[0]]=GPP_obsscale
    #obli[Obi[1]]=ET_obsscale
    obli[Obi[1]]=SWC_obsscale
    #obli[Obi[3]]=Max_obsscale
    #obli[Obi[4]]=Min_obsscale
    #obli[Obi[5]]=RO_obsscale
    
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
    print("Initial Logliklihood")
    print(Logl)
    sum_states = np.zeros(dim)
    outer_product_sum = np.zeros((dim, dim))
    num_adapted_samples = 0
    lag=0
    plus=0
    accepted_samples_for_adaptation=[]
    Trueburn=burnin+adapt_interval
    for time in list(range(1,steps)):
        
        lag=lag+1
        if time >= burnin and (time + 1) % adapt_interval == 0 and (len(accepted_samples_for_adaptation) > 1):
                # Update running sums for mean and covariance calculation
                   #print(accepted_samples_for_adaptation)
                    par_cov_matrix= (np.cov(np.array(accepted_samples_for_adaptation).T))*.5
                    print(par_cov_matrix)
                    # Add a small regularization to prevent singular covariance matrix
                    par_cov_matrix += np.eye(dim) * 1e-6
                   
    
        newproposed=np.random.multivariate_normal(np.array(x_1),par_cov_matrix, size=1).tolist()[0]
        newproposed=bounce(newproposed,pd.DataFrame(samples_scale))
        movelog=pd.concat([movelog,pd.DataFrame(newproposed)],axis=1)
        NewLogl=log_prob(regrli,scarleri,np.array(newproposed))
        NewLogl=NewLogl+plus
        scorelog=np.append(scorelog,Logl)
        if (acceptance(Logl,NewLogl)):
            Logl=NewLogl-plus
            plus=0
            lag=0
            
            x_1=newproposed
            accepted_samples_for_adaptation.append(newproposed)
            #rint('Accept')
           # print(len(accepted_samples_for_adaptation))
            if time >Trueburn:
                acceptlog=pd.concat([acceptlog,pd.DataFrame(newproposed)],axis=1)
        if lag==20:
            lag=0
            plus=plus+100
       #     x_1=newproposed
       #     accepted_samples_for_adaptation.append(newproposed)
       #     print('Accept')
       #     print(len(accepted_samples_for_adaptation))
       #     if time >Trueburn:
       #         acceptlog=pd.concat([acceptlog,pd.DataFrame(newproposed)],axis=1)
        if time % 500 == 0:
            print(time)
            print(Logl)
            print("Total Accepted")
            print(len(acceptlog.T))
            pd.DataFrame(movelog).to_csv(benchmark)

    print("Total Accepted")
    print(len(acceptlog.T))
    return(movelog)


Varset=Varset.iloc[:,1].values

indices_to_delete = np.where(Varset == "month")[0]
Varset2 = np.delete(Varset, indices_to_delete)
indices_to_delete = np.where(Varset2 == "year")[0]
Varset2 = np.delete(Varset2, indices_to_delete)

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
S=[1,1,1,1,1,1,1,1,1,10]

LOADIN=True
if LOADIN==True:
    regrli,scarleri=Loadmodels(LOADIN,Emdir,Model_List,Scaler_List)

TestLL(regrli,scarleri,initialpos=111)




par_cov_matrix=np.outer(S,S)
print(par_cov_matrix)
log=AdaptiveMCMC(par_cov_matrix,
             steps,
             adapt_interval,
             burnin,dims,
             samples_sub.iloc[[initialpos]].T,
             benchmark="bm.csv")

Fixedlog=pd.DataFrame(scaler_pars.inverse_transform(log.T))
Fixedlog.to_csv("data/MCMC_runs/TestMC.csv")