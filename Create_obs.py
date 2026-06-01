import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime

Obs=pd.DataFrame({"Date":["MeanPFT","MeanPFT","MeanPFT","MeanPFT","MeanPFT","MeanPFT","MeanPFT","MeanPFT",
                          "MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA","MeanDIA"],	
              "obs":[30,20,10,1,10,20,30,10,100,100,100,100,100,100,100,100,100,100,100,100],	
              "error":[30.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,100,100,100,100,100,100,100,100,100,100,100,100],
              "pft": [1,2,3,4,1,2,3,4,1,1,1,1,1,1,1,1,1,1,1,1],
              "case": [1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],
              "dbh":[1,1,1,1,1,1,1,1,1,2,3,4,5,6,1,2,3,4,5,6 ],
              "Set":["PFT1_BA_case1","PFT2_BA_case1","PFT3_BA_case1","PFT4_BA_case1",
                      "PFT1_BA_case2","PFT2_BA_case2","PFT3_BA_case2","PFT4_BA_case2",
                      "Stem_DIA1_case1","Stem_DIA2_case1","Stem_DIA3_case1","Stem_DIA4_case1","Stem_DIA5_case1","Stem_DIA6_case1",
                     "Stem_DIA1_case1","Stem_DIA2_case1","Stem_DIA3_case1","Stem_DIA4_case1","Stem_DIA5_case2","Stem_DIA6_case2"]
              
})
Obs["Ds"]="1990-01-01"
Obs["De"]="2000-01-01"
Obs["freq"]="YE"


df_proc=pd.DataFrame({})
for i in list(range(0,len(Obs))):
    one=Obs.iloc[[i]].reset_index()
    if ((one["Date"]=="MeanPFT")|(one["Date"]=="MeanDIA"))[0]:
        Newout=pd.DataFrame({"Year":pd.date_range(one["Ds"].values[0],one["De"].values[0],freq=one["freq"].values[0]).year})
        Newout["obs"]=np.repeat(one["obs"].values,len(Newout))
        Newout["error"]=np.repeat(one["error"].values,len(Newout))
        Newout["Set"]=np.repeat(one["Set"].values,len(Newout))
        Newout["Date"]=np.repeat(one["Date"].values,len(Newout))
        if (one["Date"]=="MeanPFT")[0]:
            Newout["pft"]=np.repeat(one["pft"].astype(int).values,len(Newout))
            Newout["case"]=np.repeat(one["case"].astype(int).values,len(Newout))       
        elif (one["Date"]=="MeanDIA")[0]:
            Newout["dbh"]=np.repeat(one["dbh"].astype(int).values,len(Newout))
            Newout["case"]=np.repeat(one["case"].astype(int).values,len(Newout))    
        df_proc=pd.concat([df_proc,Newout])
df_proc.to_csv("Xiulin_Observation.csv")