# Check if Python is installed
if (!(Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python3 could not be found. Please install Python3 and try again."
    exit 1
}

# Create a virtual environment
py -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
py -m pip install --upgrade pip

# Install the required packages
py -m pip install -r requirements.txt

Write-Host "Virtual environment has been set up and packages have been installed."
Write-Host "To activate the virtual environment, run:"
Write-Host ".\venv\Scripts\Activate.ps1"
