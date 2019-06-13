import os
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from skimage import io, color, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _fgs(model, X, y, epsilon): 
    opt = optim.Adam([X], lr=1e-3)
    X.requires_grad_()
    out = model(X)[0]
    ce = nn.CrossEntropyLoss()(out, y)
    # err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    opt.zero_grad()
    ce.backward()

    norm = torch.norm(X.grad.view(-1,torch.prod(torch.tensor(X.grad.shape[1:])).item())
                            , 2, 1)

    for i in range(norm.size(0)):
        X.grad[i,:] = X.grad[i,:] / norm[i]

    # eta = X.grad.data.sign()*epsilon
    eta = X.grad*epsilon

    X_fgs = X + eta
    # err_fgs = (model(X_fgs).data.max(1)[1] != y.data).float().sum()  / X.size(0)
    return X_fgs

def _pgd(models, X, y, epsilon, niters=100, alpha=100):

    X.requires_grad_()

    X_pgd = X
    X_pgd.requires_grad_()

    for _ in range(niters): 
        opt = optim.Adam([X_pgd], lr=1e-3)
        logit = None
        for it in models:
            if logit is None:
                logit = it["weight"]*it["model"](X_pgd/it["ratio"]) if it["mode"] != 0 else it["model"](X_pgd/it["ratio"])[0]
            else:
                logit = logit + it["weight"]*it["model"](X_pgd/it["ratio"]) if it["mode"] != 0 else it["model"](X_pgd/it["ratio"])[0]

        loss = nn.CrossEntropyLoss()(logit, y)

        opt.zero_grad()
        loss.backward()
        eta = alpha*X_pgd.grad

        X_pgd = X_pgd + eta

        # adjust to be within [-epsilon, epsilon]
        with torch.no_grad():

            eta = X_pgd - X
            norm = eta.view(eta.size(0),-1).norm(2,1)
            eta = eta.view(eta.size(0),-1)
            print(eta.size())
            for idx in range(norm.size(0)):
                if norm[idx] > epsilon:
                    eta[idx,:] = epsilon * eta[idx,:] / norm[idx].item()
            
            eta = eta.view(X.size())
            print(eta.size())
            # eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            
        
        X_pgd = torch.tensor(X + eta).requires_grad_()
        X_pgd.requires_grad_()
        '''
        if ((model(X_pgd)[0].max(1)[1] == y).float().sum()/X_pgd.size(0)).item() == 0:
            print('attack complete!')
            return X_pgd
        '''

        del eta, norm, loss
        torch.cuda.empty_cache()

    return X_pgd

def _pgd_t(models, X, t, epsilon, niters=100, alpha=100):

    X.requires_grad_()

    X_pgd = X
    X_pgd.requires_grad_()

    for _ in range(niters): 
        opt = optim.Adam([X_pgd], lr=1e-3)
        logit = None
        for it in models:
            if logit is None:
                logit = it["weight"]*it["model"](X_pgd/it["ratio"]) if it["mode"] != 0 else it["model"](X_pgd/it["ratio"])[0]
            else:
                logit = logit + it["weight"]*it["model"](X_pgd/it["ratio"]) if it["mode"] != 0 else it["model"](X_pgd/it["ratio"])[0]

        loss = nn.CrossEntropyLoss()(logit, t)

        opt.zero_grad()
        loss.backward()
        eta = alpha*X_pgd.grad

        X_pgd = X_pgd - eta

        # adjust to be within [-epsilon, epsilon]
        with torch.no_grad():

            eta = X_pgd - X
            norm = eta.view(eta.size(0),-1).norm(2,1)
            eta = eta.view(eta.size(0),-1)
            print(eta.size())
            for idx in range(norm.size(0)):
                if norm[idx] > epsilon:
                    eta[idx,:] = epsilon * eta[idx,:] / norm[idx].item()
            
            eta = eta.view(X.size())
            print(eta.size())
            # eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            
        
        X_pgd = torch.tensor(X + eta).requires_grad_()
        X_pgd.requires_grad_()
        '''
        if ((model(X_pgd)[0].max(1)[1] == y).float().sum()/X_pgd.size(0)).item() == 0:
            print('attack complete!')
            return X_pgd
        '''

        del eta, norm, loss
        torch.cuda.empty_cache()

    return X_pgd

def _mi_fgm(models, X, y, epsilon, niters=100,  momentum=1.0):

    alpha = epsilon / niters
    X.requires_grad_()
    gradient = torch.zeros(X.size()).to(device)

    for i in range(niters):
        opt = optim.Adam([X], lr=1e-3)
        logit = None
        for it in models:
            if logit is None:
                logit = it["weight"]*it["model"](X/it["ratio"]) if it["mode"] != 0 else it["model"](X/it["ratio"])[0]
            else:
                logit = logit + it["weight"]*it["model"](X/it["ratio"]) if it["mode"] != 0 else it["model"](X/it["ratio"])[0]
        
        if i == 0:
            l = logit.min(1)[1].to(device)
        loss = nn.CrossEntropyLoss()(logit, y)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():

            eta = X.grad.view(X.size(0),-1)
            norm = X.grad.view(X.size(0),-1).norm(1,1)
            for idx in range(norm.size(0)):
                eta[idx,:] = eta[idx,:] / norm[idx].item()
            eta = eta.view(X.size())
            gradient = gradient*momentum + eta

            eta = gradient.view(X.size(0),-1)
            norm = gradient.view(X.size(0),-1).norm(2,1)

            for idx in range(norm.size(0)):
                eta[idx,:] = eta[idx,:] / norm[idx].item()

            eta = eta.view(X.size())

        X = torch.tensor(X + eta*alpha)
        X.requires_grad_()
        del eta, norm, loss
        torch.cuda.empty_cache()

    return X

def _mi_fgm_t(models, X, t, epsilon, niters=100,  momentum=1.0):

    alpha = epsilon / niters
    X.requires_grad_()
    gradient = torch.zeros(X.size()).to(device)
    logits = []

    for i in range(niters):
        opt = optim.Adam([X], lr=1e-3)
        
        logit = None
        for it in models:
            if logit is None:
                logit = it["weight"]*it["model"](X/it["ratio"]) if it["mode"] != 0 else it["model"](X/it["ratio"])[0]
            else:
                logit = logit + it["weight"]*it["model"](X/it["ratio"]) if it["mode"] != 0 else it["model"](X/it["ratio"])[0]
        loss = nn.CrossEntropyLoss()(logit, t)
        '''
        logit1 = models[0]['model'](X)
        logit2 = models[1]['model'](X)
        logit3 = models[2]['model'](X)
        logit = 0.4*logit1 + 0.45*logit2 + 0.15*logit3
        loss = nn.CrossEntropyLoss()(logit, t)
        #loss = nn.CrossEntropyLoss()(logit1 ,y)*0.33 + nn.CrossEntropyLoss()(logit2, y)*0.33 + nn.CrossEntropyLoss()(logit3, y)*0.33
        '''
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():

            eta = X.grad.view(X.size(0),-1)
            norm = X.grad.view(X.size(0),-1).norm(2,1)
            for idx in range(norm.size(0)):
                eta[idx,:] = eta[idx,:] / norm[idx].item()
            eta = eta.view(X.size())
            gradient = gradient*momentum - eta

            eta = gradient.view(X.size(0),-1)
            norm = gradient.view(X.size(0),-1).norm(2,1)

            for idx in range(norm.size(0)):
                eta[idx,:] = eta[idx,:] / norm[idx].item()

            eta = eta.view(X.size())

            

        X = (torch.clamp(X + eta*alpha,0,1)).clone().detach().requires_grad_(True)
        X.requires_grad_()
        del eta, norm, loss
        torch.cuda.empty_cache()

    return X

def _mask(x, t, mask_data):
    
    x_adv = x.clone().detach()
    for idx in range(x.size(0)):
        lower = int(149 - args.mask_size/2)
        upper = int(149 + args.mask_size/2)

        x_adv[idx,:,lower:upper,
                lower:upper] = mask_data[t[idx].item()]['image']
        # print(mask_data[t[idx].item()])

    return x_adv

def _pgd_train(model, X0, X, y, epsilon, niters=100, alpha=100): 

    X.requires_grad_()

    X_pgd = X
    X_pgd.requires_grad_()

    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1e-3)

        # logit = 0.2*model1(X_pgd)[0] + 0.8*model2(X_pgd/255)
        # loss = nn.CrossEntropyLoss()(logit, y)
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        opt.zero_grad()
        loss.backward()
        eta = alpha*X_pgd.grad

        X_pgd = X_pgd + eta

        # adjust to be within [-epsilon, epsilon]
        with torch.no_grad():

            eta = X_pgd - X0
            norm = eta.view(eta.size(0),-1).norm(2,1)
            eta = eta.view(eta.size(0),-1)
            for idx in range(norm.size(0)):
                if norm[idx] > epsilon:
                    eta[idx,:] = epsilon * eta[idx,:] / norm[idx].item()
            
            eta = eta.view(X.size())
            # eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            
        
        X_pgd = torch.tensor(X0 + eta)
        X_pgd.requires_grad_()
        '''
        if ((model(X_pgd)[0].max(1)[1] == y).float().sum()/X_pgd.size(0)).item() == 0:
            print('attack complete!')
            return X_pgd
        '''

        del eta, norm, loss
        torch.cuda.empty_cache()

    return X_pgd