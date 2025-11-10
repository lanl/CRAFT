# CRAFT (Calibration and Rapid-Adoption Forecasting Techniques)

CRAFT is a Python-based project for processing, analyzing, and modeling climate or environmental data. It uses machine learning techniques, specifically Random Forest Regression, to create emulators for various environmental variables such as Gross Primary Production (GPP) and soil water content.

## Project Structure

The project consists of three main Python scripts:

1. `1_Data_Cleaning.py`: Processes and prepares input data for machine learning.
2. `2_Emulat_fitting.py`: Trains Random Forest models and performs feature importance analysis.
3. `3_Bayes_Fit.py`: Implements Bayesian inference using Markov Chain Monte Carlo (MCMC) methods for parameter estimation and uncertainty quantification.

## Dependencies

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- joblib
- scipy

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

3. Prepare your input data:
   - Ensure your input data is in CSV format.
   - Update the `Emulator_Fitting_Settings1.csv` file with the appropriate settings for your variables.

4. Run the scripts in order:
   ```
   python 1_Data_Cleaning.py
   python 2_Emulat_fitting.py
   python 3_Bayes_Fit.py
   ```

## Configuration

The project uses four main configuration files:

- `Cleaning_meta.csv`: Contains settings for the data cleaning process.
- `Emulator_Fitting_Settings1.csv`: Contains settings for the emulator fitting process.
- `Emulator_Meta.csv`: Provides metadata for the emulator process.
- `Bayes_settings.csv`: Contains settings for the Bayesian inference process.

Modify these files to adjust the variables being modeled, their input files, and other configuration parameters.

## Configuration Files

### 1. Cleaning_meta.csv (Proposed: Cleaning_Metadata_v1.csv)

Purpose: Contains settings for the data cleaning process.
Related Script: 1_Data_Cleaning.py

Key Fields:
- DRIVE: Directory containing input data
- CASE: Name of the case being processed
- N_runs: Number of runs to process
- Varlist: List of variables to be cleaned
- Parameter_loc: Location of parameter file

### 2. Emulator_Fitting_Settings1.csv (Proposed: Emulator_Settings_v1.csv)

Purpose: Contains settings for the emulator fitting process.
Related Script: 2_Emulat_fitting.py

Key Fields:
- var_name: Name of the variable being modeled (e.g., GPP)
- FATESruns: Input file for the variable data
- xs, xe: Start and end indices for data selection
- y_i: Index of the target variable
- Scaler: Scaling factor for the variable

### 3. Emulator_Meta.csv (Proposed: Emulator_Metadata_v1.csv)

Purpose: Provides metadata for the emulator process.
Related Scripts: 2_Emulat_fitting.py, 3_Bayes_Fit.py

Key Fields:
- Settings_File: Name of the emulator settings file
- SaveName: Name for saving output files
- thres: Threshold value (purpose to be clarified)
- EmulatorDrive: Directory for emulator data
- FATES_samples: Input file for FATES samples

### 4. Bayes_settings.csv (Proposed: Bayes_Settings_v1.csv)

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

To maintain consistency and clarity, we propose the following naming convention for these configuration files:

[Process]_[Purpose]_[Version].csv

For example:
- Cleaning_Metadata_v1.csv
- Emulator_Settings_v1.csv
- Emulator_Metadata_v1.csv
- Bayes_Settings_v1.csv

This naming convention clearly indicates the process, purpose, and version of each configuration file.

## Output

The project generates various outputs, including:

- Processed data files in the `data/Vars/` directory
- Trained models and scalers in the `data/Models/` directory
- Diagnostic plots in the `diag/` directory
- MCMC results in the `data/MCMC_runs/` directory

## Contributing

For information on how to contribute to this project, please refer to the CONTRIBUTING.md file (if available) or contact the project maintainers.

## License

Please refer to the LICENSE file for information on the project's license and usage terms.

## Contact

For questions or support, please contact the project maintainers or open an issue on the GitHub repository.
