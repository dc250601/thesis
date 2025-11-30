#!/bin/bash

module load tensorflow/2.12.0

srun python3 script.py --save_path ./results_GDA/B0_st_GDA --signal B0_st --mode GDA &
srun python3 script.py --save_path ./results_GDA/cas_q_hl_32_GDA --signal cas_q_hl_32 --mode GDA &
srun python3 script.py --save_path ./results_GDA/cas_c_hh_12_GDA --signal cas_c_hh_12 --mode GDA &
srun python3 script.py --save_path ./results_GDA/lambcs_bc_32_GDA --signal lambcs_bc_32 --mode GDA &
wait
