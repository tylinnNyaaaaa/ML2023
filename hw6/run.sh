#!/bin/bash
#SBATCH --account=ACD114087
#SBATCH --partition=gp1d
#SBATCH --nodes=1                           # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=1                   # Gpus per node
#SBATCH --cpus-per-task=1                   # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=4                 # Maximum number of tasks on each node
#SBATCH --time=8:00:00                      # time limit
#SBATCH --output=job-%j.out                 # (-o) Path to the standard output file
#SBATCH --error=job-%j.err                  # (-e) Path to the standard error file
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<@twcherrylin@gmail.com>   # Where to send mail.  Set this to your email address

python Unet.py

