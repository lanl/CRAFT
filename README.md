# CRAFT (Climate Research and Forecasting Tool)

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

The `Emulator_Fitting_Settings1.csv` file contains configuration settings for the emulator fitting process. Modify this file to adjust the variables being modeled, their input files, and scaling factors.

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
