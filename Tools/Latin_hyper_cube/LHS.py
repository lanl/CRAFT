# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:40:42 2024

@author: 345578
"""
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from scipy.stats import weibull_min,skewnorm,norm


Example=pd.read_csv("LHS_Range.csv")
#Example=#Example[:len(Example)-1]
n_samples=500

#dist="cauchy"
#a=-2.6780331444417245
#b=0.23711167992331433
###cauchy a=loc b=scale
def SingleLHS(n_samples,dist,a,b,c):
    if dist=="cauchy":
        divide=np.linspace(0.0001, 0.999, n_samples) ### define the spaces to sample between
        var1=np.array([])
        c=sp.cauchy(loc=a,scale=b) ### Fit distribution 
        for i in list(range(0,len(divide))):
            if i ==0:
                continue
            else: 
                out=np.random.uniform(c.ppf(divide[i-1]),c.ppf(divide[i]),size=1) ### Fit distribution 
                var1=np.append(var1,out)
                
    #dist="gamma"
    #a=11.315339532527176
    #b=-2.4991809276919588
    #c=0.6811563789574502
            
    if dist=="gamma":
        divide=np.linspace(0.01, 0.999, n_samples)
        var1=np.array([])
        c=sp.gamma(a=a,loc=b)
        for i in list(range(0,len(divide))):
            if i ==0:
                continue
            else: 
                out=np.random.uniform(c.ppf(divide[i-1]),c.ppf(divide[i]),size=1)
                var1=np.append(var1,out) 
    
    ###dist="normal"
    ### loc=mean
    ### scale=std
    if dist=="normal":
        divide=np.linspace(0.01, 0.999, n_samples)
        var1=np.array([])
        c=sp.norm(loc=a,scale=b)
        for i in list(range(0,len(divide))):
            if i ==0:
                continue
            else: 
                out=np.random.uniform(c.ppf(divide[i-1]),c.ppf(divide[i]),size=1)
                var1=np.append(var1,out)            
    ###dist="lognorm"
    # x=mean
    # s=sd
    # loc=0
    # scale=1
    if dist=="lognormal":
        divide=np.linspace(0.01, 0.999, n_samples)
        var1=np.array([])
        c=sp.lognorm(x=a,s=b,loc=a)
        for i in list(range(0,len(divide))):
            if i ==0:
                continue
            else: 
                out=np.random.uniform(c.ppf(divide[i-1]),c.ppf(divide[i]),size=1)
                var1=np.append(var1,out)             
    #dist="uniform"
    #a=1
    #b=10    
    if dist=="uniform":
        var1=np.array([])
        divide=np.linspace(a, b, n_samples)
        for i in list(range(0,len(divide))):
            if i ==0:
                continue
            else: 
                out=np.random.uniform(divide[i-1],divide[i],size=1)
                var1=np.append(var1,out)
    
    #dist="chi2"
    #a=0.850760424250364
    #b=48.71999999999999
    #c=1.3918519564774758
    if dist=="chi2":
        divide=np.linspace(0.01, 0.999, n_samples)
        var1=np.array([])
        c=sp.chi2(df=a,loc=b,scale=c)
        for i in list(range(0,len(divide))):
            if i ==0:
                continue
            else: 
                out=np.random.uniform(c.ppf(divide[i-1]),c.ppf(divide[i]),size=1)
                var1=np.append(var1,out) 
    var1=np.random.permutation(var1)
    return(var1)
outdf=pd.DataFrame({})
for tt in list(range(0,len(Example))):
    dist=Example['dist'][tt]
    a=Example['a'][tt]
    b=Example['b'][tt]
    c=Example['c'][tt]
    Outcol=SingleLHS(n_samples,dist,a,b,c)
    outdf=pd.concat([outdf,pd.DataFrame({Example['Variable'][tt]:Outcol})],axis=1)
outdf.to_csv("LHS_Samples.csv")