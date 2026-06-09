import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import shap
import random
from contextlib import redirect_stdout, redirect_stderr
import tensorflow as tf
from config_utils import get_emulator_metadata, get_emulator_settings, get_nn_config


def plotout(regr, X_test, y_test, Title, model_type="rf"):
    ### After a model is fit we will find the shapely importance of the variables and create diagnostic plots.

    if model_type == "rf":
        predicty = regr.predict(X_test)
        feature_importance = regr.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(5, 5))
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(x.columns)[sorted_idx])
        plt.title(Title+ " Feature Importance")
        t = pd.DataFrame({'x':np.array(x.columns)[sorted_idx],'b':feature_importance[sorted_idx]})
        t.loc[t["b"]>0.01]["x"].values
        plt.axvline(0.01, c="red")
        RF_testscore = regr.score(X_test, y_test)
        plt.savefig("diag/"+Title+ " Feature Importance.png", dpi=600, bbox_inches='tight')
        
        fig = plt.figure(figsize=(5, 5))
        plt.plot(predicty, y_test, 'ro', alpha=0.2, color="blue")
        plt.xlabel("predicted", fontsize=15)
        plt.ylabel("observed", fontsize=15)
        plt.title(Title+ " Testing Set", fontsize=15)
        plt.axline((min(predicty), min(predicty)), slope=1, label="1:1 line", color="red")
        plt.legend()
        plt.savefig("diag/"+Title+ " Testing Set.png", dpi=600, bbox_inches='tight')
        
        featuredf = pd.DataFrame({"Name":np.array(x.columns)[sorted_idx], "Imp":feature_importance[sorted_idx]})
        featuredf = featuredf.sort_values("Imp", ascending=False)
        print(featuredf.sort_values("Imp", ascending=False))
        return featuredf
    else:  # neural network
        # Convert to tensor format for prediction if needed
        if not isinstance(X_test, tf.Tensor):
            X_test_tensor = tf.convert_to_tensor(X_test.astype('float32'))
            predicty = regr.predict(X_test_tensor)
        else:
            predicty = regr.predict(X_test)
        #####################################################################
        # Ensure predicty is flattened for plotting
        if hasattr(predicty, "flatten"):
            predicty = predicty.flatten()
        rng = np.random.default_rng()
        print("Calculating shap values")
        X100=rng.choice(X_test, size=100, axis=0, replace=False)
        explainer = shap.Explainer(regr.predict, X100)
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                shap_values = explainer(X100,silent=True)
        feature_names=x.columns
        rf_resultX = pd.DataFrame(shap_values.values.mean(axis=0))
        #print(rf_resultX)
        vals = np.abs(rf_resultX.values)/np.abs(rf_resultX.values).sum()
        featuredf= pd.DataFrame(list(zip(feature_names, vals.flatten())),
                                  columns=['Name','Imp'])
        print(featuredf)
        featuredf.sort_values(by=['Name'],
                               ascending=False, inplace=True)
        fig = plt.figure(figsize=(10, 5))

        sorted_idx = np.argsort(vals.flatten())
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(5, 5))
        plt.barh(pos, vals.flatten()[sorted_idx], align="center")
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.savefig("diag/"+Title+ " Feature Importance.png", dpi=600, bbox_inches='tight')
    
        # Instead, we'll just plot the testing set results
        fig = plt.figure(figsize=(5, 10))
        plt.plot(predicty, y_test, 'ro', alpha=0.2, color="blue")
        plt.xlabel("predicted", fontsize=15)
        plt.ylabel("observed", fontsize=15)
        plt.title(Title+ " Testing Set (Neural Network)", fontsize=15)
        plt.axline((min(predicty), min(predicty)), slope=1, label="1:1 line", color="red")
        plt.legend()
        plt.savefig("diag/"+Title+ " Testing Set.png", dpi=600, bbox_inches='tight')
        return featuredf

def build_nn_model(input_shape, layer_sizes=[64, 32, 1], activations='relu', dropouts=0, l1=0, l2=0, optimizer='adam', loss='mse'):
    """Build a neural network model with the specified architecture"""
    def clip_above_zero(x):
        return tf.minimum(x, 0.0)
    metrics_list = ['MeanSquaredError', tf.keras.metrics.R2Score()]
    
    # Build the model
    model = tf.keras.models.Sequential()
    
    # Add input layer
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    
    # Add hidden layers (all except the last one)
    for i, size in enumerate(layer_sizes[:-1]):
        # Handle activation (could be a list or a single value)
        act = activations[i] if isinstance(activations, list) else activations
        
        # Handle regularization (could be a list or a single value)
        l1_val = l1[i] if isinstance(l1, list) else l1
        l2_val = l2[i] if isinstance(l2, list) else l2
        
        model.add(tf.keras.layers.Dense(size, activation=act, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_val, l2=l2_val)))
        
        # Handle dropout (could be a list or a single value)
        if isinstance(dropouts, list):
            drop_val = dropouts[i]
            if drop_val > 0:
                model.add(tf.keras.layers.Dropout(drop_val))
        elif dropouts and dropouts > 0:
            model.add(tf.keras.layers.Dropout(dropouts))
    
    # Add output layer (no activation for regression tasks)
    model.add(tf.keras.layers.Dense(layer_sizes[-1]))
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics_list
    )
    print(model.summary())
    return model
    
def learn(x, y, save, filename, model_type="rf", nn_config=None):
    
    scaler = StandardScaler().fit(x)
    x_z = scaler.transform(x)
    X_train, X_test, y_train, y_test = train_test_split(
             x_z, y, test_size=0.30, random_state=44)
    
    if model_type == "rf":  # Random Forest
        regr = RandomForestRegressor(n_jobs=-1).fit(X_train, y_train)
        RF_trainscore = regr.score(X_train, y_train)
        print("Training set score_ "+str(RF_trainscore))
        RF_testscore = regr.score(X_test, y_test)
        print("Test set score_ "+str(RF_testscore))
    else:  # Neural Network
        # Default NN configuration if not provided
        if nn_config is None:
            nn_config = {
                'layer_sizes': [64, 32, 1],
                'activations': ['relu', 'relu'],
                'dropouts': [0.1, 0.1],
                'l1_regularization': [0.0, 0.0],
                'l2_regularization': [0.0, 0.0],
                'early_stopping_enabled': False,
                'early_stopping_monitor': 'val_loss',
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.001,
                'optimizer': 'adam',
                'loss': 'mse',
                'epochs': 5,
                'batch_size': 32
            }
        
        # Convert to tensors for TensorFlow
        X_train_tensor = tf.convert_to_tensor(X_train.astype('float32'))
        y_train_tensor = tf.convert_to_tensor(y_train.values.reshape(-1, 1).astype('float32'))
        X_test_tensor = tf.convert_to_tensor(X_test.astype('float32'))
        y_test_tensor = tf.convert_to_tensor(y_test.values.reshape(-1, 1).astype('float32'))
        
        # Create dataset objects
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_tensor, y_test_tensor))
        
        # Configure batch size
        BATCH_SIZE = nn_config['batch_size']
        train_ds = train_ds.batch(BATCH_SIZE).cache()
        test_ds = test_ds.batch(BATCH_SIZE).cache()
        
        # Build the neural network model
        regr = build_nn_model(
            input_shape=X_train.shape[1],
            layer_sizes=nn_config['layer_sizes'],
            activations=nn_config['activations'],
            dropouts=nn_config['dropouts'],
            l1=nn_config['l1_regularization'],
            l2=nn_config['l2_regularization'],
            optimizer=nn_config['optimizer'],
            loss=nn_config['loss']
        )
        
        # Configure callbacks
        callbacks = []
        if nn_config.get('early_stopping_enabled', False):
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor=nn_config.get('early_stopping_monitor', 'val_loss'),
                patience=nn_config.get('early_stopping_patience', 10),
                min_delta=nn_config.get('early_stopping_min_delta', 0.001),
                restore_best_weights=True
            )
            callbacks.append(early_stop)

        # Train the model
        history = regr.fit(
            train_ds,
            epochs=nn_config['epochs'],
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_results = regr.evaluate(test_ds)
        train_results = regr.evaluate(train_ds)
        print(f"Training set score: {train_results}")
        print(f"Test set score: {test_results}")
    
    # Save model regardless of type
    if save==True: 
        # Create Models directory if it doesn't exist
        os.makedirs("data/Models", exist_ok=True)
        
        if model_type == "rf":
            joblib.dump(regr, "data/Models/"+filename+".joblib")
        else:
            # Add .keras extension to comply with TensorFlow's requirements
            regr.save("data/Models/"+filename+"_nn.keras")
        
        joblib.dump(scaler, "data/Models/"+filename+ "_Scalar.joblib") 
    
    return(X_train, X_test, y_train, y_test, regr)

if __name__ == "__main__":
    # Load metadata and settings
    Variables = get_emulator_settings()
    Meta = get_emulator_metadata()

    SaveName = Meta.loc[Meta["Var"]=='SaveName']['Path'].values[0]
    print(SaveName)
    thres = float(Meta.loc[Meta["Var"]=='thres']['Path'].values[0])
    EmulatorDirve = Meta.loc[Meta["Var"]=='EmulatorDrive']['Path'].values[0]
    Fates_sa = Meta.loc[Meta["Var"]=='FATES_samples']['Path']
    Fates_sa = Fates_sa.values[0]

    # Check if model_type is specified in metadata
    model_type = "rf"  # default
    if "model_type" in Meta["Var"].values:
        model_type = Meta.loc[Meta["Var"]=='model_type']['Path'].values[0]

    # Load neural network configuration if specified
    nn_config = None
    if model_type == "nn":
        try:
            nn_config_df = get_nn_config()
            nn_config = nn_config_df.iloc[0].to_dict()
            
            # Convert layer_sizes from string to list
            if isinstance(nn_config['layer_sizes'], str):
                nn_config['layer_sizes'] = [int(size) for size in nn_config['layer_sizes'].split(',')]
            
            if isinstance(nn_config['activations'], str):
                nn_config['activations'] = [act.strip() for act in nn_config['activations'].split(',')]
            
            if isinstance(nn_config['dropouts'], str):
                nn_config['dropouts'] = [float(d.strip()) for d in nn_config['dropouts'].split(',')]
            
            if isinstance(nn_config['l1_regularization'], str):
                nn_config['l1_regularization'] = [float(l.strip()) for l in nn_config['l1_regularization'].split(',')]
            
            if isinstance(nn_config['l2_regularization'], str):
                nn_config['l2_regularization'] = [float(l.strip()) for l in nn_config['l2_regularization'].split(',')]
            
            print(f"Neural network configuration loaded from XML config")
        except Exception as e:
            print(f"Failed to load neural network configuration from XML: {e}")

    # Create diagnostic directory if it doesn't exist
    os.makedirs("diag", exist_ok=True)

    # Load FATES samples
    FATES_samples = pd.read_csv(Fates_sa)

    ### Initial full feature importance analysis
    Full = True
    if Full==True:
        ImportanceDF = pd.DataFrame({})
        for i in list(range(0,len(Variables["xs"]))):
            xs = Variables.loc[i,"xs"]
            xe = Variables.loc[i,"xe"]
            y_i = Variables.loc[i,"y_i"]
            var_name = Variables.loc[i,"var_name"]
            Downsample=Variables.loc[i,"downsample"]
            Scaler = Variables.loc[i,"Scaler"]
            GPP = pd.read_csv(EmulatorDirve+Variables.loc[i, 'FATESruns'])
            print("Creating full emulator for")

            GPP.sample(n=int(len(GPP)*Downsample))

            x = GPP.iloc[:,np.r_[xs:xe]] ###here skipping all but the first tminus
            y = GPP.iloc[:,y_i]*Scaler
            print(var_name)
            print("Xstart XEnd Yvariable")
            print(xs,xe,y_i)
            X_train, X_test, y_train, y_test, regr = learn(x, y, True, SaveName+var_name, model_type, nn_config)
            featuredf = plotout(regr, X_test, y_test, var_name, model_type)
            print("Done")
      
            row = pd.DataFrame({'Predicting': var_name,
                            'Variable': featuredf.loc[featuredf["Imp"]>thres]["Name"].values,
                            "Importance": featuredf.loc[featuredf["Imp"]>thres]["Imp"].values})
            ImportanceDF = pd.concat([ImportanceDF, row])
        
        # Save importance data if it exists (only for RF)
        if len(ImportanceDF) > 0:
            ImportanceDF.to_csv("diag/Full_Importance.csv")
            NewVars = pd.DataFrame(ImportanceDF["Variable"].unique())
            NewVars.to_csv("diag/FitOrder.csv")
        else:
            # For NN, we don't have feature importance, so create an empty file
            pd.DataFrame().to_csv("diag/Full_Importance.csv")
            # Use all columns for reduced model
            NewVars = pd.DataFrame(x.columns)
            NewVars.to_csv("diag/FitOrder.csv")
    if thres==0:
        exit()
    # For the second stage (reduced model), load the fit order from previous stage
    if os.path.exists("diag/FitOrder.csv"):
         # Random Forest
            NewVars = pd.read_csv("diag/FitOrder.csv")
            NewVars = NewVars.iloc[:,1].values
            print(NewVars)
            ImportanceDF = pd.DataFrame({}) 
            for i in list(range(0, len(Variables["xs"]))):
                y_i = Variables.loc[i,"y_i"]
                var_name = Variables.loc[i,"var_name"]
                Scaler = Variables.loc[i,"Scaler"]
                GPP = pd.read_csv(EmulatorDirve+Variables.loc[i, 'FATESruns'])
                #print(GPP.columns)
                print("Creating an emulator with reduced parameters for")
                # Make sure all NewVars exist in the dataframe
                valid_cols = [col for col in NewVars if col in GPP.columns]
                x = GPP[valid_cols]
                
                x = x[valid_cols]
                y = GPP.iloc[:,y_i]*Scaler
                #print(x)
                print(var_name)
                X_train, X_test, y_train, y_test, regr = learn(x, y, True, SaveName+var_name, model_type, nn_config)

                
                featuredf = plotout(regr, X_test, y_test, var_name, model_type)
                row = pd.DataFrame({'Predicting': var_name,
                            'Variable': featuredf["Name"].values,
                            "Importance": featuredf["Imp"].values})
                ImportanceDF = pd.concat([ImportanceDF, row])
            
            # Save importance data if it exists 
            if len(ImportanceDF) > 0:
                ImportanceDF.to_csv("diag/Reduced_Importance.csv")
