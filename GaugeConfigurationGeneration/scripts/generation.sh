#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --array=1-8%1
#SBATCH --requeue

# Get the start time for the entire script
script_start=$(date +%s)

export MPICH_GPU_SUPPORT_ENABLED=0

PATH_SCRIPT='/global/u1/d/diptarko/lumin/milc_qcd-7.7.13/ks_imp_dyn/su3_rmd'

PATH_INPUT='/pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/inputs'
PATH_OUTPUT='/pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/logs'
    
# Get the index from the Python script
idx=$(python3 /pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/scripts/index.py)

# Modify input/output paths based on the index
PATH_INPUT="${PATH_INPUT}/in${idx}"
PATH_OUTPUT="${PATH_OUTPUT}/out${idx}"

# Create the input script
python3 /pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/scripts/input-generator.py $idx $PATH_INPUT
    
# Run the job
srun -n 512 "$PATH_SCRIPT" < "$PATH_INPUT" > "$PATH_OUTPUT"


