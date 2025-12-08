# CRAFT (Calibration and Rapid-Adoption Forecasting Techniques)

CRAFT is a Python-based project for processing, analyzing, and modeling climate or environmental data. It uses machine learning techniques, including Random Forest Regression and Neural Networks, to create emulators for various environmental variables such as Gross Primary Production (GPP) and soil water content.

## Project Structure

The project consists of three main Python scripts:

1. `1_Data_Cleaning.py`: Processes and prepares input data for machine learning.
2. `2_Emulat_fitting.py`: Trains machine learning models (Random Forest or Neural Network) and performs feature analysis.
3. `3_Bayes_Fit.py`: Implements Bayesian inference using Markov Chain Monte Carlo (MCMC) methods for parameter estimation and uncertainty quantification.

## Dependencies

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- joblib
- scipy
- tensorflow (for neural network models)

You can install these dependencies using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Setup and Usage

1. Clone the repository:
   ```
   git clone https://github.com/lanl/CRAFT.git
   cd CRAFT
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   Alternatively, you can use the provided setup script:
   ```
   python setup_venv.py
   ```

3. Prepare your input data:
   - Ensure your input data is in CSV format.
   - Update the `Emulator_Settings_v1.csv` file with the appropriate settings for your variables.
   - Choose your model type (Random Forest or Neural Network) in the `Emulator_Metadata_v1.csv` file.

4. Run the scripts in order:
   ```
   python 1_Data_Cleaning.py
   python 2_Emulat_fitting.py
   python 3_Bayes_Fit.py
   ```

## Configuration

The project uses five main configuration files:

- `Cleaning_Metadata_v1.csv`: Contains settings for the data cleaning process.
- `Emulator_Settings_v1.csv`: Contains settings for the emulator fitting process.
- `Emulator_Metadata_v1.csv`: Provides metadata for the emulator process.
- `Bayes_Settings_v1.csv`: Contains settings for the Bayesian inference process.
- `nn_config.csv`: Contains configuration for neural network models (if selected).

Modify these files to adjust the variables being modeled, their input files, model type, and other configuration parameters.

## Configuration Files

### 1. Cleaning_Metadata_v1.csv

Purpose: Contains settings for the data cleaning process.
Related Script: 1_Data_Cleaning.py

Key Fields:
- DRIVE: Directory containing input data
- CASE: Name of the case being processed
- N_runs: Number of runs to process
- Varlist: List of variables to be cleaned
- Parameter_loc: Location of parameter file

### 2. Emulator_Settings_v1.csv

Purpose: Contains settings for the emulator fitting process.
Related Script: 2_Emulat_fitting.py

Key Fields:
- var_name: Name of the variable being modeled (e.g., GPP)
- FATESruns: Input file for the variable data
- xs, xe: Start and end indices for data selection
- y_i: Index of the target variable
- Scaler: Scaling factor for the variable

### 3. Emulator_Metadata_v1.csv

Purpose: Provides metadata for the emulator process.
Related Scripts: 2_Emulat_fitting.py, 3_Bayes_Fit.py

Key Fields:
- Settings_File: Name of the emulator settings file
- SaveName: Name for saving output files
- thres: Threshold value for feature importance
- EmulatorDrive: Directory for emulator data
- FATES_samples: Input file for FATES samples
- model_type: Type of model to use ('rf' for Random Forest or 'nn' for Neural Network)
- nn_config: Path to the neural network configuration file (used only when model_type='nn')

### 4. Bayes_Settings_v1.csv

Purpose: Contains settings for the Bayesian inference process.
Related Script: 3_Bayes_Fit.py

Key Fields:
- initialpos: Initial position for the MCMC algorithm
- steps: Number of MCMC steps
- adapt_interval: Adaptation interval for the MCMC algorithm
- burnin: Number of burn-in steps
- Model_List: List of models to use
- Scaler_List: List of scalers to use
- Obs_List: List of observations to use
- list1: Additional list of parameters (purpose to be clarified)
- samples_file: File containing samples for the MCMC process
- obsfile: File containing observation data

### 5. nn_config.csv

Purpose: Contains configuration for neural network models.
Related Script: 2_Emulat_fitting.py (when model_type='nn')

Key Fields:
- layer_sizes: Comma-separated list of layer sizes (e.g., "64,32,1")
- activation: Activation function for hidden layers (e.g., 'relu')
- optimizer: Optimizer for training (e.g., 'adam')
- loss: Loss function (e.g., 'mse')
- epochs: Number of training epochs
- batch_size: Batch size for training

## Model Types

### Random Forest (model_type='rf')
- Default model type
- Provides feature importance analysis
- Works well with smaller datasets
- Less computationally intensive
- Generates diagrams showing feature importance

### Neural Network (model_type='nn')
- Alternative model type for complex relationships
- Requires more data for effective training
- May capture more complex patterns
- Configurable through nn_config.csv
- Does not provide feature importance analysis

## Output

The project generates various outputs, including:

- Processed data files in the `data/Vars/` directory
- Trained models and scalers in the `data/Models/` directory
  - Random Forest models are saved with .joblib extension
  - Neural Network models are saved in directories with _nn suffix
- Diagnostic plots in the `diag/` directory
- MCMC results in the `data/MCMC_runs/` directory

## Contributing

For information on how to contribute to this project, please refer to the CONTRIBUTING.md file (if available) or contact the project maintainers.

## License

Please refer to the LICENSE file for information on the project's license and usage terms.

## Contact

For questions or support, please contact the project maintainers or open an issue on the GitHub repository.
