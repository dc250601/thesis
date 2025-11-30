import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool


import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts as CAWR
from util import *

import argparse

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--signal",type=str,required=True)
    parser.add_argument("--mode",type=str,required=True)

    
    args = parser.parse_args()

    save_path = args.save_path
    signal = args.signal
    mode = args.mode

    DATA_PATH = "../../../CorrelatorStudy/Bdata/ext2"
    f = open(f"{DATA_PATH}/{signal}", "r")
    data = f.readlines()
    f.close()


    mean_energy_list = []
    std_energy_list = []
    raw_energy_list = []
    collapsed_energy_list = []
    cut = []
    for i in tqdm(range(0,23)):
        cut.append(i)
        CUT = i
        ################################################################################################
        y = np.array(list(map(lambda x: abs(float(x.split("\t")[-1].split("\n")[0])), data)), dtype=float)
        x = np.array(list(map(lambda x: float(x.split("\t")[0].split("\n")[0]), data)), dtype=float)
        
        idx = (x>=CUT)
        
        x = x[idx]
        y = y[idx]
        
        mean,std = GD_with_error(x=x,
                                y=y,
                                mode=mode,
                                Nprocess=128,
                                Nproc=128
                                               )
        mean_energy_list.append(mean)
        std_energy_list.append(std)
    
    
    GS = np.array(mean_energy_list)[:,0]
    E1 = np.array(mean_energy_list)[:,1]
    
    GS_error = np.array(std_energy_list)[:,0]
    E1_error = np.array(std_energy_list)[:,1]

    np.save(save_path+"_std",GS_error)
    np.save(save_path+"_mean",GS)
    