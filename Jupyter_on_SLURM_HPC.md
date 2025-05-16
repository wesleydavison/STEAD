# Running Jupyter Notebook on a SLURM HPC Cluster

This guide walks you through running Jupyter Notebook on a SLURM-based HPC cluster, from setup to shutdown.

---

## 1. Prepare Your Conda Environment (One-Time Setup)

**a.** Log in to the cluster and activate your environment:
```bash
conda activate eqt
```

**b.** Make sure Jupyter is installed:
```bash
conda install -y jupyter
```

---

## 2. Create or Update Your SLURM Script

Create a file called `jupyter_slurm.sh` with the following content (adjust paths if needed):

```bash
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

# (Optional) Start in your project directory
cd ~/STEAD

# Get the compute node hostname and a random port
export NODE_HOSTNAME=$(hostname -s)
export PORT=$(shuf -i 8000-9000 -n 1)

echo "Starting Jupyter on node: $NODE_HOSTNAME"
echo "Using port: $PORT"
echo "Jupyter path: $(which jupyter)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0
```

---

## 3. Submit the SLURM Job

```bash
sbatch jupyter_slurm.sh
```

---

## 4. Check Job Status and Get Connection Info

```bash
squeue -u $USER
```
Find your job ID (e.g., 14893).

Check the log file for the port and token:
```bash
cat jupyter_<jobid>.log
```
Look for lines like:
- `Using port: 8514`
- A URL with `?token=...`

---

## 5. Set Up SSH Tunnel (on Your Local Machine)

Open a terminal on your local computer and run:
```bash
ssh -N -L <PORT>:<NODE_HOSTNAME>:<PORT> 230442014@nandi.campus.aston.ac.uk
```
Replace `<PORT>` and `<NODE_HOSTNAME>` with values from your log file.

---

## 6. Open Jupyter in Your Browser

Go to:
```
http://localhost:<PORT>
```
Paste the token from your log file if prompted.

---

## 7. Work in Jupyter!

- Navigate to your files and run your notebooks as needed.

---

## 8. Shut Down When Done

**a.** In Jupyter, shut down any running notebooks/kernels.
**b.** On the login node, cancel your job:
```bash
scancel <jobid>
```
**c.** Close your SSH tunnel or VSCode Remote session.

---

## 9. (Optional) Clean Up Temporary Files

If you created a special directory for Jupyter, you can remove it:
```bash
rm -rf ~/jupyter_runtime
```

---

## Summary

You're now running Jupyter on a SLURM compute node, using the cluster's resources safely and efficiently! 