#!/bin/bash

module load tensorflow/2.12.0

srun -n 1 --exclusive python3 script.py --save_path ./results_GD/B0_st_GD --signal B0_st --mode GD &
srun -n 1 --exclusive python3 script.py --save_path ./results_GD/cas_q_hl_32_GD --signal cas_q_hl_32 --mode GD &
srun -n 1 --exclusive python3 script.py --save_path ./results_GD/cas_c_hh_12_GD --signal cas_c_hh_12 --mode GD &
srun -n 1 --exclusive python3 script.py --save_path ./results_GD/lambcs_bc_32_GD --signal lambcs_bc_32 --mode GD &
wait
