import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path


lu=pd.read_csv("Cleaning_Metadata_v1.csv")
print(lu)
DRIVE1=lu["DRIVE"][0]
print(DRIVE1)

CASENAME=lu["CASE"][0]
print(CASENAME)
varlist=lu["Varlist"].values[0]
varlist=varlist.split(",")
print(varlist)
Nruns=int(lu["N_runs"].values)
print(Nruns)
samples=lu["Parameter_loc"].values[0]
print(samples)


#varlist=["Date",'FATES_GPP']
samples=pd.read_csv(samples)


print("Loading in files:")
print(str(DRIVE1+CASENAME+str(1)+".csv"))

df=pd.DataFrame()
for i in list(range(0,Nruns)):
    my_file=DRIVE1+CASENAME+str(i)+".csv"
    if os.path.isfile(my_file):
        print(i)
        one=pd.read_csv(my_file) 
        one=one[varlist]   
        one['Date']=pd.DatetimeIndex(one["Date"])
        #rint(one["Date"])
        one['Month']=one["Date"].dt.month
        one['Year']=one["Date"].dt.year
        #one['Date2']=str(one['Date'])
        two=one.groupby(["Month","Year"],as_index=False).mean()
        cols=varlist.copy()
        cols.extend(["Month","Year"])
        print(cols)
        two=two[cols]
        #two["Date"]=pd.DatetimeIndex(two['Year'].astype(str)+"-"+two['Month'].astype(str)+"-01")
#       two=two[["Date",'FATES_GPP']]
        two["Step"]=list(range(0,len(two)))
        two["Model"]=i
        df=pd.concat([df,two])

print(df)
print("Writing out variables for training:")
samples["Model"]=list(range(1,len(samples)+1))

for var in varlist[1:]:
   
    df2=pd.DataFrame()
    ms=df["Model"].unique()
    for i in ms: 
        print(i)
        checky=df[["Date","Step",var,"Model"]]
        sub1=checky[checky["Model"]==i]
        #ub1['Date']=pd.DatetimeIndex(sub1["Date"])
        
        sub1=sub1.iloc[4:,:].merge(samples,on="Model",how="left")
        sub1=sub1.reset_index(drop=True)
        

        sub1['Date']=pd.DatetimeIndex(sub1["Date"])
        sub1["month"]=sub1["Date"].dt.month
        sub1["year"]=sub1["Date"].dt.year
        
        df2=pd.concat([df2,sub1])
        
    df2.to_csv('data/Vars/'+var+'forML.csv')

#Qsoil=pd.read_csv('C:/Users/345578/Desktop/NewML/FATES_outputs/QSOIL11425.csv')
#QVEGE=pd.read_csv('C:/Users/345578/Desktop/NewML/FATES_outputs/QVEGE11425.csv')
#QVEGT=pd.read_csv('C:/Users/345578/Desktop/NewML/FATES_outputs/QVEGT11425.csv')
