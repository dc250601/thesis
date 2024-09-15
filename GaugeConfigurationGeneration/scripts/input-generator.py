import numpy as np
import os
import sys
import shutil
from tqdm.auto import tqdm
import glob

############################################################################################################
############################################################################################################
CONFIG_PATH = "/pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/configuration"
OUTPUT_PATH = "/pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/logs/"
INPUT_PATH = "/pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/inputs/"
############################################################################################################
############################################################################################################
config = {
    "nflavors1": 2,
    "nflavors2": 1,
    "nx": 16,
    "ny": 16,
    "nz": 16,
    "nt": 48,
    "iseed": 5682304,
    "warms": 2000, #Should be 2000 by default
    "trajecs": 125000,
    "traj_between_meas": 1,
    "beta": 6.572,
    "mass1": 0.0097,
    "mass2": 0.0484,
    "u0": 1,
    "microcanonical_time_step": 0.02,
    "steps_per_trajectory": 5,
    "max_cg_iterations": 10000,
    "max_cg_restarts": 2,
    "error_per_site": 0.000001,
    "error_for_propagator": 0.000002,
    "npbp_reps": 1,
    "prec_pbp": 1,}

############################################################################################################
############################################################################################################

def raw_script(config, index=0):
    
    """
    configs: These are the default values which is there in the input script given the script runs infinitely
    index: The index is the new index from which the input script will be created. 0 is when the script begins and warmups are started. nth is the nth configuration after warmup.
    """
    
    nx = config["nx"]
    ny = config["ny"]
    nz = config["nz"]
    nt = config["nt"]
    nflavors1 = config["nflavors1"]
    nflavors2 = config["nflavors2"]
    iseed = config["iseed"]
    warms = config["warms"]
    trajecs = config["trajecs"]
    traj_between_meas = config["traj_between_meas"]
    beta = config["beta"]
    mass1 = config["mass1"]
    mass2 = config["mass2"]
    u0 = config["u0"]
    microcanonical_time_step = config["microcanonical_time_step"]
    steps_per_trajectory = config["steps_per_trajectory"]
    max_cg_iterations = config["max_cg_iterations"]
    max_cg_restarts = config["max_cg_restarts"]
    error_per_site = config["error_per_site"]
    error_for_propagator = config["error_for_propagator"]
    npbp_reps = config["npbp_reps"]
    prec_pbp = config["prec_pbp"]
    
    
    intial_input = f"""prompt 0
nflavors1 {nflavors1}
nflavors2 {nflavors2}
nx {nx}
ny {ny}
nz {nz}
nt {nt}
iseed {iseed}"""

    
    warm_input = f"""
    
warms {warms}
trajecs {0}
traj_between_meas {2}
beta {beta}
mass1 {mass1}
mass2 {mass2}
u0 {u0}
microcanonical_time_step {microcanonical_time_step}
steps_per_trajectory {steps_per_trajectory}
max_cg_iterations {max_cg_iterations}
max_cg_restarts {max_cg_restarts}
error_per_site {error_per_site}
error_for_propagator {error_for_propagator}
npbp_reps {npbp_reps}
prec_pbp {prec_pbp}
fresh
"""

    iter_input = f"""
    
warms {0}
trajecs {1}
traj_between_meas {1}
beta {beta}
mass1 {mass1}
mass2 {mass2}
u0 {u0}
microcanonical_time_step {microcanonical_time_step}
steps_per_trajectory {steps_per_trajectory}
max_cg_iterations {max_cg_iterations}
max_cg_restarts {max_cg_restarts}
error_per_site {error_per_site}
error_for_propagator {error_for_propagator}
npbp_reps {npbp_reps}
prec_pbp {prec_pbp}
"""
    complete_input = ""
    
    if index == 0:
        save_input = f"""save_serial {os.path.join(CONFIG_PATH,"conf0")}"""
        complete_input = intial_input + warm_input + save_input ## Warmup part complete
    else:
        complete_input = intial_input
        
    for t in range(index,trajecs):
        io_input = f"""reload_serial {os.path.join(CONFIG_PATH,f"conf{t}")}
save_serial {os.path.join(CONFIG_PATH,f"conf{t+1}")}"""
        complete_input = complete_input + iter_input + io_input

    
    return complete_input

def save_text(input_text,path):
    f = open(path,"w")
    f.write(input_text)
    

if __name__ == "__main__":
    
    # Will be using sys.argv since this is anyway a helper script and is not intended to run by someone else
    
    index = int(sys.argv[1])
    path_input = sys.argv[2]
    
    input_ = raw_script(config = config, index = index)
    save_text(input_text = input_, path = path_input)
    