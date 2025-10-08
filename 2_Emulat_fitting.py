import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def plotout(regr,X_test,y_test,Title):
    predicty=regr.predict(X_test)
    feature_importance = regr.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(5, 5))
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(x.columns)[sorted_idx])
    plt.title(Title+ " Feature Importance")
    t=pd.DataFrame({'x':np.array(x.columns)[sorted_idx],'b':feature_importance[sorted_idx]})
    t.loc[t["b"]>0.01]["x"].values
    plt.axvline(0.01,c="red")
    RF_testscore = regr.score(X_test, y_test)
    plt.savefig("diag/"+Title+ " Feature Importance.png",dpi=600, bbox_inches='tight')
    #plt.show()
    
    fig = plt.figure(figsize=(5, 5))
    plt.plot(predicty,y_test,'ro',alpha=0.2,color="blue")
    plt.xlabel("predicted", fontsize=15)
    plt.ylabel("observed", fontsize=15)
    plt.title(Title+ " Testing Set", fontsize=15)
    plt.axline((min(predicty), min(predicty)), slope=1,label="1:1 line",color="red")
    plt.legend()
    plt.savefig("diag/"+Title+ " Testing Set.png",dpi=600, bbox_inches='tight')
   # plt.show()
    featuredf=pd.DataFrame({"Name":np.array(x.columns)[sorted_idx],"Imp":feature_importance[sorted_idx]})
    featuredf=featuredf.sort_values("Imp",ascending=False)
    print(featuredf.sort_values("Imp",ascending=False))
    return(featuredf)
    
def learn(x,y,save,filename):
    scaler =StandardScaler().fit(x)
    x_z=scaler.transform(x)
    X_train, X_test, y_train, y_test = train_test_split(
             x_z, y, test_size=0.30, random_state=44)
    regr = RandomForestRegressor(n_jobs=-1).fit(X_train, y_train)
    RF_trainscore = regr.score(X_train, y_train)
    print("Training set score_ "+str(RF_trainscore))
    RF_testscore = regr.score(X_test, y_test)
    print("Test set score_ "+str(RF_testscore))
    if save==True: 
        joblib.dump(regr, "data/Models/"+filename+".joblib") 
        joblib.dump(scaler, "data/Models/"+filename+ "_Scalar.joblib") 
    return(X_train, X_test, y_train, y_test,regr)

Variables=pd.read_csv("C:/Users/345578/Documents/GitHub/CRAFT/Emulator_Fitting_Settings1.csv")
SaveName="CRAFT_practice_Full"
thres=.01
EmulatorDirve="C:/Users/345578/Desktop/CRAFT_Data/FATES_outputs/"
Downsample=1.0

FATES_samples=pd.read_csv('C:/Users/345578/Desktop/CRAFT_Data/LHS.sam.csv')

### This is the csv you load in: 
Full=False
if Full==True:
    ImportanceDF=pd.DataFrame({})
    for i in list(range(0,len(Variables["xs"]))):
        xs=Variables.loc[i,"xs"]
        xe=Variables.loc[i,"xe"]
        y_i=Variables.loc[i,"y_i"]
        var_name=Variables.loc[i,"var_name"]
        Scaler=Variables.loc[i,"Scaler"]
        GPP=pd.read_csv(EmulatorDirve+Variables.loc[i, 'FATESruns'])
        if var_name in ["LWP_max"]:
            GPP["DOY"]=pd.DatetimeIndex(GPP['Date']).dayofyear
            GPP["Year"]=pd.DatetimeIndex(GPP['Date']).year
       # else: 
       #     GPP["month"]=pd.DatetimeIndex(GPP['Date']).month
       #     GPP["Year"]=pd.DatetimeIndex(GPP['Date']).year
        x=GPP.iloc[:,np.r_[xs:xe]] ###here skipping all but the first tminus
        y=GPP.iloc[:,y_i]*Scaler
        print(var_name)
        X_train, X_test, y_train, y_test,regr=learn(x,y,True,SaveName+var_name)
        featuredf=plotout(regr,X_test,y_test,var_name)
        row=pd.DataFrame({'Predicting':var_name,
                    'Variable':featuredf.loc[featuredf["Imp"]>thres]["Name"].values,
                    "Importance":featuredf.loc[featuredf["Imp"]>thres]["Imp"].values})
        ImportanceDF=pd.concat([ImportanceDF,row])
    ImportanceDF.to_csv("diag/Full_Importance.csv")
    NewVars=pd.DataFrame(ImportanceDF["Variable"].unique())
    NewVars.to_csv("diag/FitOrder.csv")



NewVars=pd.read_csv("diag/FitOrder.csv")
NewVars=NewVars.iloc[:,1].values
print(NewVars)
ImportanceDF=pd.DataFrame({})
for i in list(range(0,len(Variables["xs"]))):
    y_i=Variables.loc[i,"y_i"]
    var_name=Variables.loc[i,"var_name"]
    Scaler=Variables.loc[i,"Scaler"]
    GPP=pd.read_csv(EmulatorDirve+Variables.loc[i, 'FATESruns'])
    print(GPP.columns)
    x=GPP[NewVars]
    #x.loc[:,"month"]=GPP["month"]
    #x.loc[:,"year"]=GPP["year"]
    x=x[NewVars]
     ###here skipping all but the first tminus
    #if var_name in ["LWP_max"]:
    #    GPP["DOY"]=pd.DatetimeIndex(GPP['Date']).dayofyear
    #    GPP["Year"]=pd.DatetimeIndex(GPP['Date']).year
    #else: 
    #    GPP["month"]=pd.DatetimeIndex(GPP['Date']).month
    #    GPP["Year"]=pd.DatetimeIndex(GPP['Date']).year
    y=GPP.iloc[:,y_i]*Scaler
    print(x)
    print(var_name)
    X_train, X_test, y_train, y_test,regr=learn(x,y,True,SaveName+var_name)
    featuredf=plotout(regr,X_test,y_test,var_name)
    row=pd.DataFrame({'Predicting':var_name,
                   'Variable':featuredf["Name"].values,
                   "Importance":featuredf["Imp"].values})
    ImportanceDF=pd.concat([ImportanceDF,row])
ImportanceDF.to_csv("diag/Reduced_Importance.csv")