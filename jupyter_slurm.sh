#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_%j.log
#SBATCH --error=jupyter_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nodes
#SBATCH --qos=normal

# Initialize conda for batch jobs
source /opt/gridware/depots/92d57679/el8/pkg/apps/miniconda3/24.4.0/etc/profile.d/conda.sh
conda activate eqt

# Create a directory for the notebook
mkdir -p ~/jupyter_runtime
cd ~/STEAD

# Get the compute node hostname
export NODE_HOSTNAME=$(hostname -s)

# Get a random port number between 8000-9000
export PORT=$(shuf -i 8000-9000 -n 1)

# Print some debug information
echo "Starting Jupyter on node: $NODE_HOSTNAME"
echo "Using port: $PORT"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Jupyter path: $(which jupyter)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Start Jupyter Notebook
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0

# Print connection information
echo "To connect to the Jupyter Notebook, run the following command on your local machine:"
echo "ssh -N -L $PORT:$NODE_HOSTNAME:$PORT $USER@nandi.campus.aston.ac.uk"
echo "Then open your browser and go to: http://localhost:$PORT"

squeue -u $USER 