import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path


DRIVE1='C:/Users/345578/outputs925/outputtest_Teller_Sens_run.IELMFATES/'
CASENAME="Teller_Sens_run.IELMFATES"
Nruns=500
varlist=["Date",'FATES_GPP','H2OSOIvalue1']
samples=pd.read_csv('C:/Users/345578/Desktop/CRAFT_Data/LHS.sam.csv')


print("Loading in files:")
df=pd.DataFrame()
for i in list(range(0,Nruns)):
    print(i)
    my_file=DRIVE1+CASENAME+str(i)+".csv"
    if os.path.isfile(my_file):
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

#print(df)
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