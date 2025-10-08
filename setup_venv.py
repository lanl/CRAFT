import venv
import subprocess
import sys
import os

def create_venv():
    venv.create('venv', with_pip=True)

def install_requirements():
    if sys.platform == 'win32':
        pip_path = os.path.join('venv', 'Scripts', 'pip')
    else:
        pip_path = os.path.join('venv', 'bin', 'pip')
    
    subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])

if __name__ == '__main__':
    create_venv()
    install_requirements()
    print("Virtual environment created and requirements installed successfully.")
