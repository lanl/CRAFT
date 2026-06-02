# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 18:43:19 2023

@author: 345578
"""
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

w_dir='' #where is the folder
fast_para=pd.read_csv(w_dir+"LHS_Samples.csv").iloc[:,1:]### here is an example
input_para = pd.read_csv(w_dir+"PARAM_Settings.csv")
#the input fates parameter file name
#nfname.in = "fates_params_default_056193a_mod1PFTs_pureEDbareground_exp4.nc"  
nfname_in = w_dir+"fates_params_deciduous_shrubs_graminoids_c20250402.nc"  
#the start of sample index
sample_start_index = 1
#the end of sample index
sample_end_index = 500
Tempname='Sens_PARAMS_' #### What will we name the new parameterfiles
#### format being Tempname.# in iteration ex CA_chap.1,CA_chap.2


###Functions
def conversion(y,l,u):
  x=+y*(u-l)
  return(x)
#assign values for the ncfile
def assign_fates_para(fates_para_values,fast_val,pft_dim_lower,
                      pft_dim_upper,tissue_dim_lower,tissue_dim_upper):
    if(len(fates_para_values))>1:
        if (pft_dim_lower or tissue_dim_lower==0):
            print("The pft/tissue dimsion should be more than 0 for "+str(var_name),"!")
            pass
        fates_para_values [pft_dim_lower:pft_dim_upper,tissue_dim_lower:tissue_dim_upper]<-fast_val 
    else:
            if (pft_dim_lower>0 or tissue_dim_lower>0):
                print("Error: Both pft and tissue dimsion are more than 0 for "+str(var_name)+"!\n"+'Set the pft indices (pft_start	and pft_end) to zero if only one pft is present!')
            if pft_dim_lower>0:
                fates_para_values[pft_dim_lower:pft_dim_upper]<-fast_val
            if tissue_dim_lower>0:
                fates_para_values [tissue_dim_lower:tissue_dim_upper]<-fast_val
            if pft_dim_lower==0 and tissue_dim_lower==0:
                fates_para_values<-fast_val
    
    
    return (fates_para_values)
#500
# Input parameters 
#
########Setting up 
n_par=len(input_para.index)
para_OI_fates=input_para['parameter']
pft_dim_lower=input_para['pft.start']
pft_dim_upper=input_para['pft.end']
tissue_dim_lower=input_para['org.start']
tissue_dim_upper=input_para['org.end']
reverse_flag=input_para['reverse.flag']
reciprocal_flag=input_para['reciprocal.flag']
conversion_flag=input_para['conversion.flag']
conversion_lower=input_para['conversion.lower']
conversion_upper=input_para['conversion.upper']
scale=input_para['scale']
shift=input_para['shift']
print(n_par)
print(len(fast_para.columns))
if(n_par!=len(fast_para.columns)):
  print("The FATES parameter setting does not match the FATES input file") 


### Testing Sturcture

test_flag=True

if(test_flag==True):
  fates_para=xr.open_dataset(nfname_in)
  list_var =fates_para.var
  #write.table(file="varname.txt",list_var)
  var_name="fates_hydro_pinot_node"
  fates_para[var_name].units
  fates_para[var_name].long_name
  para_values=fates_para[var_name].values
  fates_para[var_name].dims
###Make copies
fates_para_values=fates_para[var_name]
for i in list(range(sample_start_index,sample_end_index+1)):
  filename1=xr.open_dataset(nfname_in)
  filename2=w_dir+'PARAMS/'+Tempname+str(i)+".nc"
  filename1.to_netcdf(filename2,mode="w")
  filename1.close()
  filename2=xr.open_dataset(filename2)
  filename2.close()

for i in list(range(sample_start_index,sample_end_index+1)):
   # filename1=xr.open_dataset(w_dir+'Outputs/NMSBA_Sens_'+str(i)+".nc")
    filename1=xr.open_dataset(nfname_in)
    print(i)
    for j in list(range(0,n_par)):   
            
            var_name=para_OI_fates[j]
            print(var_name)
            if pd.notna(var_name):
                fast_val=(fast_para.iloc[i,j]+shift.iloc[j])*1
            if reverse_flag[j]==1: fast_val=-fast_val;
            if reciprocal_flag[j]==1: fast_val=1.0/fast_val;
            if conversion_flag[j]==1: fast_val=conversion(fast_val,conversion_lower[j],conversion_upper[j])
            fates_para_values=filename1[var_name]
            
           # if ['fates_leafage_class','fates_pft'] in fates_para_values.dims:
           #     fates_para_values.values[0,pft_dim_lower[j]-1]=fast_val
            
            if 'fates_pft' in fates_para_values.dims:
                if 'fates_leafage_class' in fates_para_values.dims:
                    fates_para_values.values[0,pft_dim_lower[j]-1]=fast_val
                elif('fates_hydr_organs' in fates_para_values.dims):
                    fates_para_values.values[tissue_dim_lower[j]-1,pft_dim_lower[j]-1]=fast_val
                else: 
                    fates_para_values.values[pft_dim_lower[j]-1]=fast_val  #### double check this.
       
            
            
            
            else:
                fates_para_values=fast_val
    print(datetime.now())
    filename1.to_netcdf(w_dir+'PARAMS/'+Tempname+str(i)+".nc")
    filename1.close()
