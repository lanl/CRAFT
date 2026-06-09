import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from config_utils import get_cleaning_metadata

os.makedirs("data", exist_ok=True)
os.makedirs("data/Vars", exist_ok=True)
lu=get_cleaning_metadata()
print(lu)
DRIVE1=lu["DRIVE"][0]
print(DRIVE1)

CASENAME=lu["CASE"][0]
print(CASENAME)

varlist=lu["Varlist"].values[0]
varlist=varlist.split(",")
print(varlist)
Nruns=int(lu["N_runs"].values[0])
print(Nruns)
samples=lu["Parameter_loc"].values[0]
print(samples)
timesteps=lu["timestep"].values[0]
timesteps=timesteps.split(",")
#varlist=["Date",'FATES_GPP']
samples=pd.read_csv(samples)

##############################################################################
############Month ############################################################
##############################################################################
subset_varlist = [
    item for item, category in zip(varlist, timesteps) if category == "Month"
]
print("Processing Monthly Values")
print("Monthly Variable")
print(subset_varlist)
print("Loading in files:")
print(str(DRIVE1+CASENAME+str(1)+".csv"))
###################Process Monthly Variables
df=pd.DataFrame()
for i in list(range(0,Nruns)):
    my_file=DRIVE1+CASENAME+str(i)+".csv"
    #print(my_file)
    if os.path.isfile(my_file):
        #print(i)
        one=pd.read_csv(my_file) 
        one=one[subset_varlist]   
        one['Date']=pd.DatetimeIndex(one["Date"])
        one['Month']=one["Date"].dt.month
        one['Year']=one["Date"].dt.year
        two=one.groupby(["Month","Year"],as_index=False).mean()
        cols=subset_varlist.copy()
        cols.extend(["Month","Year"])
        #print(cols)
        two=two[cols]
        two["Step"]=list(range(0,len(two)))
        two["Model"]=i
        df=pd.concat([df,two])

print(df)
print("Writing out variables for training:")
samples["Model"]=list(range(1,len(samples)+1))

for var in subset_varlist[1:]:
    print(var)
    df2=pd.DataFrame()
    ms=df["Model"].unique()
    for i in ms: 
       
        checky=df[["Date","Step",var,"Model"]]
        sub1=checky[checky["Model"]==i]
        sub1=sub1.iloc[4:,:].merge(samples,on="Model",how="left")
        sub1=sub1.reset_index(drop=True)
        sub1['Date']=pd.DatetimeIndex(sub1["Date"])
        sub1["month"]=sub1["Date"].dt.month
        sub1["year"]=sub1["Date"].dt.year
        df2=pd.concat([df2,sub1])
    df2.to_csv('data/Vars/'+var+'MonthlyforML.csv')

####################################################################################
############Daily Mean  ############################################################
####################################################################################


print("Processing Daily Values")
subset_varlist = [
    item for item, category in zip(varlist, timesteps) if category == "Day"
]
subset_varlist.insert(0, "Date")
print("Variables for Daily")
print(subset_varlist)
print("Loading in files:")
print(str(DRIVE1+CASENAME+str(1)+"h1.csv"))

df=pd.DataFrame()
for i in list(range(0,Nruns)):
    my_file=DRIVE1+CASENAME+str(i)+"h1.csv"
    #print(my_file)
    if os.path.isfile(my_file):
        #print(i)
        one=pd.read_csv(my_file) 
        one=one[subset_varlist]   
        one['Date']=pd.DatetimeIndex(one["Date"])
        one['DOY']=one["Date"].dt.dayofyear
        one['Year']=one["Date"].dt.year
        two=one.groupby(['DOY',"Year"],as_index=False).mean()
        cols=subset_varlist.copy()
        cols.extend(['DOY',"Year"])
        #print(cols)
        two=two[cols]
        two["Step"]=list(range(0,len(two)))
        two["Model"]=i
        df=pd.concat([df,two])

#print(df)
print("Writing out variables for training:")
samples["Model"]=list(range(1,len(samples)+1))

for var in subset_varlist[1:]:
    print(var)
    df2=pd.DataFrame()
    ms=df["Model"].unique()
    for i in ms: 
        
        checky=df[["Date","Step",var,"Model"]]
        sub1=checky[checky["Model"]==i]
        sub1=sub1.iloc[4:,:].merge(samples,on="Model",how="left")
        sub1=sub1.reset_index(drop=True)
       # sub1['Date']=pd.DatetimeIndex(sub1["Date"])
        sub1['DOY']=sub1["Date"].dt.dayofyear
        sub1["year"]=sub1["Date"].dt.year
        df2=pd.concat([df2,sub1])
    df2.to_csv('data/Vars/'+var+'DailyforML.csv')


##############################################################################
############################Daily Min#########################################
##############################################################################

subset_varlist = [
    item for item, category in zip(varlist, timesteps) if category == "DayMin"
]
subset_varlist.insert(0, "Date")

print("Variables for DayMin" )
print(subset_varlist)
print("Loading in files:")
print(str(DRIVE1+CASENAME+str(1)+"h1.csv"))

df=pd.DataFrame()
for i in list(range(0,Nruns)):
    my_file=DRIVE1+CASENAME+str(i)+"h1.csv"
    #print(my_file)
    if os.path.isfile(my_file):
        #print(i)
        one=pd.read_csv(my_file) 
        one=one[subset_varlist]   
        one['Date']=pd.DatetimeIndex(one["Date"])
        one['DOY']=one["Date"].dt.dayofyear
        one['Year']=one["Date"].dt.year
        two=one.groupby(['DOY',"Year"],as_index=False).min()
        cols=subset_varlist.copy()
        cols.extend(['DOY',"Year"])
        #print(cols)
        two=two[cols]
        two["Step"]=list(range(0,len(two)))
        two["Model"]=i
        df=pd.concat([df,two])

#print(df)
print("Writing out variables for training:")
samples["Model"]=list(range(1,len(samples)+1))

for var in subset_varlist[1:]:
    print(var)
    df2=pd.DataFrame()
    ms=df["Model"].unique()
    for i in ms: 
        
        checky=df[["Date","Step",var,"Model"]]
        sub1=checky[checky["Model"]==i]
        sub1=sub1.iloc[4:,:].merge(samples,on="Model",how="left")
        sub1=sub1.reset_index(drop=True)
       # sub1['Date']=pd.DatetimeIndex(sub1["Date"])
        sub1['DOY']=sub1["Date"].dt.dayofyear
        sub1["year"]=sub1["Date"].dt.year
        df2=pd.concat([df2,sub1])
    df2.to_csv('data/Vars/'+var+'DailyminforML.csv')

    subset_varlist = [
    item for item, category in zip(varlist, timesteps) if category == "DayMax"
]
#subset_varlist.insert(0, "Date")

print(subset_varlist)
print("Loading in files:")

df=pd.DataFrame()
for i in list(range(0,Nruns)):
    my_file=DRIVE1+CASENAME+str(i)+"h1.csv"
    #print(my_file)
    if os.path.isfile(my_file):
        #print(i)
        one=pd.read_csv(my_file) 
        one=one[subset_varlist]   
        one['Date']=pd.DatetimeIndex(one["Date"])
        one['DOY']=one["Date"].dt.dayofyear
        one['Year']=one["Date"].dt.year
        two=one.groupby(['DOY',"Year"],as_index=False).max()
        cols=subset_varlist.copy()
        cols.extend(['DOY',"Year"])
        #print(cols)
        two=two[cols]
        two["Step"]=list(range(0,len(two)))
        two["Model"]=i
        df=pd.concat([df,two])

#print(df)
print("Writing out variables for training:")
samples["Model"]=list(range(1,len(samples)+1))
print(subset_varlist)
for var in subset_varlist[1:]:
    print(var)
    df2=pd.DataFrame()
    ms=df["Model"].unique()
    for i in ms: 
        
        checky=df[["Date","Step",var,"Model"]]
        sub1=checky[checky["Model"]==i]
        sub1=sub1.iloc[4:,:].merge(samples,on="Model",how="left")
        sub1=sub1.reset_index(drop=True)
       # sub1['Date']=pd.DatetimeIndex(sub1["Date"])
        sub1['DOY']=sub1["Date"].dt.dayofyear
        sub1["year"]=sub1["Date"].dt.year
        df2=pd.concat([df2,sub1])
    df2.to_csv('data/Vars/'+var+'DailymaxforML.csv')


##################################################
##############By PFT Annual Max ##################
##################################################
print("Variables for PFT Annual Max")

subset_varlist = [
    item for item, category in zip(varlist, timesteps) if category == "PFTannMax"
]
print(subset_varlist)
for var in subset_varlist[0:]:
    pdf3=pd.read_csv(DRIVE1+CASENAME+".csv")
    #print(pdf3)
    pdf3=pdf3[["ens_label","case","pft" ,"time",var]]
    Fullstack=pdf3.merge(samples, on=["ens_label","case"],how='left')
    Fullstack["Year"]=Fullstack["time"].str.split('T', expand=True)[0].str.split('-', expand=True)[0]
    Fullstack=Fullstack.loc[Fullstack["Year"].astype(int)>1950] ### add date cutoff function. 
    Fullstack2=Fullstack.groupby(['ens_label','case', 'pft',"Year"],as_index=False).max()
    cols_to_move = [var,'ens_label',"time","case","pft" ,"Year"]
    X_data = Fullstack2[cols_to_move + [col for col in Fullstack2.columns if col not in cols_to_move]]  

    X_data.to_csv('data/Vars/'+var+'PFTannMean.csv')

##################################################
##############By DIA Annual Max ##################
##################################################
print("Variables for DIA Annual Max")

subset_varlist = [
    item for item, category in zip(varlist, timesteps) if category == "DIAannMax"
]

print(subset_varlist)
for var in subset_varlist[0:]:
    pdf3=pd.read_csv(DRIVE1+"TwoSites-PreIndust-AllParams-ens_bydbh-001.csv")
    pdf3=pdf3[["ens_label","case","dbh","time",var]]
    Fullstack=pdf3.merge(samples, on=["ens_label","case"],how='left')
    Fullstack["Year"]=Fullstack["time"].str.split('T', expand=True)[0].str.split('-', expand=True)[0]
    Fullstack=Fullstack.loc[Fullstack["Year"].astype(int)>1950]
    Fullstack2=Fullstack.groupby(['ens_label','case', 'dbh',"Year"],as_index=False).max()
    cols_to_move = [var,'ens_label',"time","case","dbh" ,"Year"]
    X_data = Fullstack2[cols_to_move + [col for col in Fullstack2.columns if col not in cols_to_move]]
    

    X_data.to_csv('data/Vars/'+var+'PFTannMean.csv')