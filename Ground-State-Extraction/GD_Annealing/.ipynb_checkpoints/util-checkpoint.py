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
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts as CAWR


class Estimator(nn.Module):
    
    def __init__(self,order):
        super(Estimator,self).__init__()
        self.E = torch.nn.Parameter(torch.abs(torch.randn(order)), requires_grad=True)
        self.A = torch.nn.Parameter(torch.abs(torch.randn(order)), requires_grad=True)
    
    def forward(self, x):
        term_exp = torch.exp(-1*x[:,None]*self.E) 
        term = torch.sum(self.A*term_exp,dim=-1)
        return term



def train_step(x,log_y,model,optimiser):
    for param in model.parameters():
        param.grad = None
    y_pred = model(x)
    y_pred_log = torch.log(y_pred)
    loss = torch.mean((y_pred_log - log_y)**2)
    loss.backward()
    optimiser.step()

def trainer(model,optimiser,scheduler,x,y,NEpochs,batch_size,verbose=True):
    model.train()
    
    if verbose:
        iterator = tqdm(range(0,NEpochs,1))
    else:
        iterator = range(0,NEpochs,1)
    
    log_y = torch.log(y)
    
    for present_epoch in iterator:
        train_loss = 0
        with torch.no_grad():
            idx = torch.randperm(x.shape[0])
            x = x[idx]
            log_y = log_y[idx]

        for i in range(0,x.shape[0]//batch_size,1):
            fit_x = x[i*batch_size:(i+1)*batch_size]
            fit_y_log = log_y[i*batch_size:(i+1)*batch_size]
            train_step(x=fit_x,log_y=fit_y_log,model=model,optimiser=optimiser)
        
        if scheduler != None:
            scheduler.step()
    return model


def GD_Annealing(data):
    
    seed = data["seed"]

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    x = data["x"]
    y = data["y"]
    
    idx = np.random.choice(a = np.arange(0,x.shape[0]),size = x.shape[0])
    x = x[idx]
    y = y[idx]
    
    device = "cpu"
    batch_size = 256
    
    model = Estimator(2)
    
    optimiser = torch.optim.Adam(model.parameters(),lr = 1e-3)
    
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    
    scheduler = CAWR(optimiser,
                     first_cycle_steps=500,
                     cycle_mult=1.0,
                     max_lr=1e-3,
                     min_lr=0,
                     warmup_steps=10,
                     gamma=0.8)
    
    model.train()

    trainer(model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            x=x_tensor,
            y=y_tensor,
            # NEpochs=1000,
            NEpochs=10000,
            batch_size=256,
            verbose=False)

    with torch.no_grad():

        y_pred = model(x_tensor)
        y_pred_log = torch.log(y_pred)
        loss = torch.mean((y_pred_log - torch.log(y_tensor))**2)

    
    return np.array(model.E.detach().numpy().tolist() + [loss.numpy()])[:,None]



def GD(data):
    
    seed = data["seed"]

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    x = data["x"]
    y = data["y"]
    
    idx = np.random.choice(a = np.arange(0,x.shape[0]),size = x.shape[0])
    x = x[idx]
    y = y[idx]
    
    device = "cpu"
    batch_size = 256
    
    model = Estimator(2)
    
    optimiser = torch.optim.Adam(model.parameters(),lr = 5e-4)
    
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    
    model.train()

    trainer(model=model,
            optimiser=optimiser,
            scheduler=None,
            x=x_tensor,
            y=y_tensor,
            NEpochs=10000,
            # NEpochs=1000,
            batch_size=256,
            verbose=False)

    with torch.no_grad():

        y_pred = model(x_tensor)
        y_pred_log = torch.log(y_pred)
        loss = torch.mean((y_pred_log - torch.log(y_tensor))**2)
    
    return np.array(model.E.detach().numpy().tolist() + [loss.numpy()])[:,None]


def GD_with_error(x,y,mode,Nproc = 250,Nprocess = 250):
    
    if mode=="GD":
        fit_func = GD
    elif mode=="GDA":
        fit_func=GD_Annealing
    else:
        print(mode,"is not valid")
    
    Nproc = Nproc
    Nprocess = Nprocess
    pool = Pool(Nproc)
    master_ss = np.random.SeedSequence(12345).spawn(Nprocess)
    seeds = [int(ss.generate_state(1)[0]) for ss in master_ss]
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    log_y = np.log(y)
    reg = LinearRegression(fit_intercept=True,n_jobs=1).fit(x.reshape(-1, 1), log_y)
    y_pred = reg.predict(x.reshape(-1, 1))
    error_se = np.mean((log_y-y_pred)**2)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    results = pool.map(fit_func,[{"x":x,"y":y,"order":2,"seed":seeds[i]} for i in range(Nprocess)])
    
    
    
    results = np.concatenate(results,-1)
    # print(results.shape)
    
    e = results[:-1,:].T
    error = results[-1,:]


    idx = error < error_se
    e = e[idx]
    e = np.abs(e)
    
    e = np.sort(e,axis=1)
    
    mean_e = np.mean(e,axis = 0)
    std_e = np.std(e,axis=0)
        
    return mean_e, std_e