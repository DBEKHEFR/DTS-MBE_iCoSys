#!/bin/bash
#SBATCH --job-name=my_python_job   # Job name
#SBATCH --output=output.log         # Output file name
#SBATCH --error=error.log           # Error file name
#SBATCH --partition=gpu             # Partition to submit to
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=1           # Number of CPUs per task
#SBATCH --gres=gpu:v100_16gb:1       # Request a specific GPU
#SBATCH --time=00:15:00             # Time limit hrs:min:sec

# Check if uv is installed, if not, install it
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing uv..."
    # Replace this with the actual command to install uv
    # For example, if uv is installable via pip, you might use:
    pip install uv  # Ensure this command is correct for your setup
fi

# Create a virtual environment using uv
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install your dependencies
uv pip install pandas numpy matplotlib scikit-learn

# Run your Python script here (assuming it's in the same directory)
python GP_test.py

# Deactivate the environment
deactivate

