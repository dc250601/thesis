import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
import numpy as np

class Guess(nn.Module):
    
    def __init__(self,feature_size,mul,nparam):
        super(Guess,self).__init__()
        
        self.layer1 = nn.Linear(feature_size,feature_size*mul)
        self.batchnorm1 = nn.BatchNorm1d(feature_size*mul)
        self.layer2 = nn.Linear(feature_size*mul,feature_size*mul)
        self.batchnorm2 = nn.BatchNorm1d(feature_size*mul)
        self.layer3 = nn.Linear(feature_size*mul,feature_size*mul)
        self.batchnorm3 = nn.BatchNorm1d(feature_size*mul)
        
        self.energy_layer= nn.Linear(feature_size*mul,nparam//2)
        self.coeff_layer= nn.Linear(feature_size*mul,nparam//2)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.LeakyReLU(x)
        
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)

        x_energy = self.ReLU(self.energy_layer(x))
        x_coeff = self.ReLU(self.coeff_layer(x))
        return x_energy, x_coeff


def train_guess_network(model,x_tensor,y_tensor,NEpochs,optimizer,BATCH_SIZE,noise_dim,device,verbose=True):

    model.train()

    
    log_y = torch.log(y_tensor)
    
    if verbose:
        pbar = tqdm(range(0, NEpochs), desc="Training", leave=True)
    else:
        pbar = range(0,NEpochs)
    for present_epoch in pbar:
        train_loss = 0
        val_loss = 0
        train_steps = 0
        test_steps = 0
        noise = torch.randn(BATCH_SIZE,noise_dim,device=device)
        x_energy,x_coeff = model(noise)
        
        for param in model.parameters():
            param.grad = None
    
    
        idx = torch.randperm(x_tensor.shape[0])[:BATCH_SIZE]
        
        exp1 = x_coeff[:,0][:,None]*torch.exp(-1*x_energy[:,0][:,None]@x_tensor[None,idx])
        exp2 = x_coeff[:,1][:,None]*torch.exp(-1*x_energy[:,1][:,None]@x_tensor[None,idx])
        y_pred_log = torch.log(exp1 + exp2+2e-100)
        fitting_loss = torch.mean((y_pred_log - log_y[None,idx])**2)
        
        std1 = torch.sqrt(x_energy.var(dim=0))
        std2 = torch.sqrt(x_coeff.var(dim=0))
        std_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
    
        loss = fitting_loss + std_loss 
        loss.backward()
        optimizer.step()

        if verbose:
        
            pbar.set_postfix({
                    'Total Loss': f'{loss.item():.4f}',
                    'Fit': f'{fitting_loss.item():.4f}',
                    'Var': f'{std_loss.item():.4f}'
                })
    return model


def double_exponential(t, E0,E1,a,b):
    return np.log(a * np.exp(-E0 * t) + b * np.exp(-E1 * t))
    

def iterate_through_noise(x,
                          y,
                          nsamples,
                          noise_dim,
                          model,
                          fit_func):

    noise = torch.randn(nsamples,noise_dim)
   
    with torch.no_grad():
        e,c = model(noise)
        e = e.to("cpu")
        c = c.to("cpu")
        
    param_NG = []
    param_B = []

    for i in range(noise.shape[0]):
        g = e[i,:].tolist() + c[i,:].tolist()
        n = noise[i,:4].tolist()
        ############################################################################## 
        try:
            popt_double_NG, cov = curve_fit(fit_func, x, np.log(y), p0=g, method="lm")
            param_NG.append((popt_double_NG[:2],np.mean((np.log(y) - fit_func(x,*popt_double_NG))**2)))
        except:
            continue
        ############################################################################## 
        try:
            popt_double_Baseline, cov2 = curve_fit(fit_func, x, np.log(y), p0=n, method="lm")
            param_B.append((popt_double_Baseline[:2],np.mean( (np.log(y) - fit_func(x,*popt_double_Baseline))**2)))
        except:
            continue
            
    return param_B, param_NG
    