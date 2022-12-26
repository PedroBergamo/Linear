import sys
import subprocess

packages = ["pandas", "scikit-learn", "sklearn", "matplotlib", "pytest"]



# implement pip as a subprocess:
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
