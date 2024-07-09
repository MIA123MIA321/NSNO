# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:00:32 2022

@author: c
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import scipy.io as scio
import time
import torch.distributed as dist
from functools import reduce


def SEED_SET(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def load_data(filename, times, operator, f_real, device):
    
    data = np.load(filename)
    k = data['k']*1
    N_sample = data['N_sample']*1
    N = data['s']*1//times
    # comp_s = data['comp_s']*1
    # times = data['times']*1
    
    # coef_q = data['coef_q']
    # coef_f = data['coef_f']
    q = data['q'][:, ::times, ::times]
    f = data['f'][:, ::times, ::times]
    u = data['exact'][:, ::times, ::times]
    ini = data['ini'][:, ::times, ::times]
    
    q_torch = torch.from_numpy(q).float().unsqueeze(1).to(device)
    
    if f_real:
        f_torch = torch.from_numpy(f).float().unsqueeze(1).to(device)
    else:
        f_real = torch.from_numpy(f.real).float()
        f_imag = torch.from_numpy(f.imag).float()
        f_torch = torch.stack((f_real, f_imag), dim=1).to(device)
            
    if operator == 'f':
        x = f_torch
    else:
        x = torch.cat((q_torch, f_torch), dim=1)
    
    # ini_real = torch.from_numpy(ini.real).float()
    # ini_imag = torch.from_numpy(ini.imag).float()
    # ini = torch.stack((ini_real, ini_imag), dim=1).to(device)
    
    u_real = torch.from_numpy(u.real).float()
    u_imag = torch.from_numpy(u.imag).float()
    y = torch.stack((u_real, u_imag), dim=1).to(device)
    
    if operator == 'f':
        # return k, N_sample, N, x, ini, ini
        return k, N_sample, N, x, ini
    else:
        # return k, N_sample, N, x, y, ini
        return k, N_sample, N, x, y

def load_data_mnist(filename, device):
    
    data = np.load(filename)

    q_train = data['q_train']
    q_test = data['q_test']
    f = data['f']
    u_train = data['exact_train']
    u_test = data['exact_test']
    
    N_train = q_train.shape[0]
    N_test = q_test.shape[0]
    
    q_train = torch.from_numpy(q_train).float().unsqueeze(1).to(device)
    q_test = torch.from_numpy(q_test).float().unsqueeze(1).to(device)
    
    f_real = torch.from_numpy(f.real).float()
    f_imag = torch.from_numpy(f.imag).float()
    f = torch.stack((f_real, f_imag), dim=1).to(device)
    
    u_train_real = torch.from_numpy(u_train.real).float()
    u_train_imag = torch.from_numpy(u_train.imag).float()
    u_train = torch.stack((u_train_real, u_train_imag), dim=1).to(device)
    
    u_test_real = torch.from_numpy(u_test.real).float()
    u_test_imag = torch.from_numpy(u_test.imag).float()
    u_test = torch.stack((u_test_real, u_test_imag), dim=1).to(device)
    
    return N_train, N_test, q_train, q_test, f, u_train, u_test

def train_model_PINN(train_loader, test_loader, model, epochs, optimizer, scheduler, N, k, operator, data_weight, PINN_weight):
    train_er = []
    test_er = []
    PINN_train_er = []
    PINN_test_er = []
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    for ep in range(epochs):
        t1 = time.time()
        train_l2 = 0.0
        PINN_train_l2 = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)

            res = torch.sqrt(torch.sum((out-y)**2, dim=(1, 2, 3)))
            ynorm = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
            loss1 = torch.sum(res/ynorm)
            
            if operator == 'f':
                res = Helm_res(out[:,0,:,:]+1j*out[:,1,:,:], N, x[:,0,:,:], k)
            else:
                res = Helm_q_res(out[:,0,:,:]+1j*out[:,1,:,:], N, x[:,1,:,:], x[:,0,:,:], k)
            PINN_res = torch.sqrt(torch.sum((torch.abs(res))**2, dim=(1, 2)))/N
            loss2 = torch.sum(PINN_res/ynorm)
            
            loss = loss1*data_weight + loss2*PINN_weight
            loss.backward()

            optimizer.step()
            train_l2 += loss1.item()
            PINN_train_l2 += loss2.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        PINN_test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:

                out = model(x)

                res = torch.sqrt(torch.sum((out-y)**2, dim=(1, 2, 3)))
                ynorm = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
                loss1 = torch.sum(res/ynorm)
                
                test_l2 += loss1.item()

                if operator == 'f':
                    res = Helm_res(out[:,0,:,:]+1j*out[:,1,:,:], N, x[:,0,:,:], k)
                else:
                    res = Helm_q_res(out[:,0,:,:]+1j*out[:,1,:,:], N, x[:,1,:,:], x[:,0,:,:], k)
                PINN_res = torch.sqrt(torch.sum((torch.abs(res))**2, dim=(1, 2)))/N
                loss2 = torch.sum(PINN_res/ynorm)
                PINN_test_l2 += loss2.item()

        train_l2/= n_train
        PINN_train_l2/= n_train
        test_l2 /= n_test
        PINN_test_l2 /= n_test

        t2 = time.time()
        if (ep + 1) % 10 == 0 or ep == 0:
            print ('Epoch [{}/{}], train: {:.4f}, PINN_train: {:.4f}, test: {:.4g}, PINN_test: {:.4g}, Time per epoch: {:.4f}'.format(ep+1, epochs, train_l2, PINN_train_l2, test_l2, PINN_test_l2, t2 - t1))

        train_er.append(train_l2)
        test_er.append(test_l2)
        PINN_train_er.append(PINN_train_l2)
        PINN_test_er.append(PINN_test_l2)
    
    return train_er, test_er, PINN_train_er, PINN_test_er

def train_model_PINN_fc(train_loader, test_loader, model, epochs, optimizer, scheduler, N, k, f, operator, data_weight, PINN_weight):
    train_er = []
    test_er = []
    PINN_train_er = []
    PINN_test_er = []
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    for ep in range(epochs):
        t1 = time.time()
        train_l2 = 0.0
        PINN_train_l2 = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)

            qf = -k**2*x*f

            res = torch.sqrt(torch.sum((out-y)**2, dim=(1, 2, 3)))
            ynorm = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
            loss1 = torch.sum(res/ynorm)
            
            if operator == 'f':
                res = Helm_res(out[:,0,:,:]+1j*out[:,1,:,:], N, x[:,0,:,:], k)
            else:
                res = Helm_q_res(out[:,0,:,:]+1j*out[:,1,:,:], N, qf[:,0,:,:]+1j*qf[:,1,:,:], x[:,0,:,:], k)
            PINN_res = torch.sqrt(torch.sum((torch.abs(res))**2, dim=(1, 2)))/N
            loss2 = torch.sum(PINN_res/ynorm)
            
            loss = loss1*data_weight + loss2*PINN_weight
            loss.backward()

            optimizer.step()
            train_l2 += loss1.item()
            PINN_train_l2 += loss2.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        PINN_test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:

                out = model(x)
                qf = -k**2*x*f

                res = torch.sqrt(torch.sum((out-y)**2, dim=(1, 2, 3)))
                ynorm = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
                loss1 = torch.sum(res/ynorm)
                
                test_l2 += loss1.item()

                if operator == 'f':
                    res = Helm_res(out[:,0,:,:]+1j*out[:,1,:,:], N, x[:,0,:,:], k)
                else:
                    res = Helm_q_res(out[:,0,:,:]+1j*out[:,1,:,:], N, qf[:,0,:,:]+1j*qf[:,1,:,:], x[:,0,:,:], k)
                PINN_res = torch.sqrt(torch.sum((torch.abs(res))**2, dim=(1, 2)))/N
                loss2 = torch.sum(PINN_res/ynorm)
                PINN_test_l2 += loss2.item()

        train_l2/= n_train
        PINN_train_l2/= n_train
        test_l2 /= n_test
        PINN_test_l2 /= n_test

        t2 = time.time()
        if (ep + 1) % 10 == 0 or ep == 0:
            print ('Epoch [{}/{}], train: {:.4f}, PINN_train: {:.4f}, test: {:.4g}, PINN_test: {:.4g}, Time per epoch: {:.4f}'.format(ep+1, epochs, train_l2, PINN_train_l2, test_l2, PINN_test_l2, t2 - t1))

        train_er.append(train_l2)
        test_er.append(test_l2)
        PINN_train_er.append(PINN_train_l2)
        PINN_test_er.append(PINN_test_l2)
    
    return train_er, test_er, PINN_train_er, PINN_test_er

def Laplace_res(x, N):
    xtop = x[:, 0:-2, 1:-1]
    xleft = x[:, 1:-1, 0:-2]
    xbottom = x[:, 2:, 1:-1]
    xright = x[:, 1:-1, 2:]
    xcenter = x[:, 1:-1, 1:-1]
    return (xtop + xleft + xbottom + xright - 4*xcenter)*N**2

def Helm_res(x, N, f, k):
    x = F.pad(x, (1,1,1,1))
    x[:, 0, 1:-1] = 2*k/N*1j*x[:, 1, 1:-1] + x[:, 2, 1:-1]
    x[:, -1, 1:-1] = 2*k/N*1j*x[:, -2, 1:-1] + x[:, -3, 1:-1]
    x[:, 1:-1, 0] = 2*k/N*1j*x[:, 1:-1, 1] + x[:, 1:-1, 2]
    x[:, 1:-1, -1] = 2*k/N*1j*x[:, 1:-1, -2] + x[:, 1:-1, -3]
    res = Laplace_res(x, N)+k**2*x[:, 1:-1, 1:-1]-f
    return res

def Helm_q_res(x, N, f, q, k):
    x = F.pad(x, (1,1,1,1))
    x[:, 0, 1:-1] = 2*k/N*1j*x[:, 1, 1:-1] + x[:, 2, 1:-1]
    x[:, -1, 1:-1] = 2*k/N*1j*x[:, -2, 1:-1] + x[:, -3, 1:-1]
    x[:, 1:-1, 0] = 2*k/N*1j*x[:, 1:-1, 1] + x[:, 1:-1, 2]
    x[:, 1:-1, -1] = 2*k/N*1j*x[:, 1:-1, -2] + x[:, 1:-1, -3]
    res = Laplace_res(x, N)+k**2*(1+q)*x[:, 1:-1, 1:-1]-f
    return res

def model_test(train_loader, test_loader, model):
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    
    train_l2 = 0.0
    with torch.no_grad():
        for x, y in train_loader:
            
            device = next(model.parameters()).device
            x = x.to(device)
            y = y.to(device)
            
            out = model(x)

            res = torch.sqrt(torch.sum((out-y)**2, dim=(1, 2, 3)))
            ynorm = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
            loss = torch.sum(res/ynorm)

            train_l2 += loss.item()

    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            
            device = next(model.parameters()).device
            x = x.to(device)
            y = y.to(device)
            
            out = model(x)

            res = torch.sqrt(torch.sum((out-y)**2, dim=(1, 2, 3)))
            ynorm = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
            loss = torch.sum(res/ynorm)

            test_l2 += loss.item()

    train_l2/= n_train
    test_l2 /= n_test
    
    return train_l2, test_l2

def train_model_mnist(train_loader, test_loader, model, epochs, optimizer, scheduler, N, k, f, data_weight, PINN_weight):
    train_er = []
    test_er = []
    PINN_train_er = []
    PINN_test_er = []
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    for ep in range(epochs):
        t1 = time.time()
        train_l2 = 0.0
        PINN_train_l2 = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            
            Nf = f.shape[0]
            Batch_size = x.shape[0]
            loss1 = 0
            loss2 = 0
            for nf in range(Nf):
                fc = f[nf].unsqueeze(0).repeat(Batch_size,1,1,1)
                qf = -k**2*x*fc
                ipt = torch.cat((x, qf), dim=1)
                out = model(ipt)
                
                res = torch.sqrt(torch.sum((out-y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                ynorm = torch.sqrt(torch.sum((y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                loss1 += torch.sum(res/ynorm)/Nf

                res = Helm_q_res(out[:,0,:,:]+1j*out[:,1,:,:], N, qf[:,0,:,:]+1j*qf[:,1,:,:], x.squeeze(1), k)
                PINN_res = torch.sqrt(torch.sum((torch.abs(res))**2, dim=(1, 2)))/N
                loss2 += torch.sum(PINN_res/ynorm)/Nf
            
            loss = loss1*data_weight + loss2*PINN_weight
            loss.backward()

            optimizer.step()
            train_l2 += loss1.item()
            PINN_train_l2 += loss2.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        PINN_test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                
                Nf = f.shape[0]
                Batch_size = x.shape[0]
                loss1 = 0
                loss2 = 0
                
                for nf in range(Nf):
                    fc = f[nf].unsqueeze(0).repeat(Batch_size,1,1,1)
                    qf = -k**2*x*fc
                    ipt = torch.cat((x, qf), dim=1)
                    out = model(ipt)

                    res = torch.sqrt(torch.sum((out-y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                    ynorm = torch.sqrt(torch.sum((y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                    loss1 += torch.sum(res/ynorm)/Nf

                    res = Helm_q_res(out[:,0,:,:]+1j*out[:,1,:,:], N, qf[:,0,:,:]+1j*qf[:,1,:,:], x.squeeze(1), k)
                    PINN_res = torch.sqrt(torch.sum((torch.abs(res))**2, dim=(1, 2)))/N
                    loss2 += torch.sum(PINN_res/ynorm)/Nf
                
                test_l2 += loss1.item()
                PINN_test_l2 += loss2.item()

        train_l2/= n_train
        PINN_train_l2/= n_train
        test_l2 /= n_test
        PINN_test_l2 /= n_test

        t2 = time.time()
        if ep == 0 or (ep + 1) % 10 == 0:
            print ('Epoch [{}/{}], train: {:.4f}, PINN_train: {:.4f}, test: {:.4g}, PINN_test: {:.4g}, Time per epoch: {:.4f}'.format(ep+1, epochs, train_l2, PINN_train_l2, test_l2, PINN_test_l2, t2 - t1))

        train_er.append(train_l2)
        test_er.append(test_l2)
        PINN_train_er.append(PINN_train_l2)
        PINN_test_er.append(PINN_test_l2)
    
    return train_er, test_er, PINN_train_er, PINN_test_er

def model_test_mnist(train_loader, test_loader, model, k, f):
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    
    train_l2 = 0.0
    with torch.no_grad():
        for x, y in train_loader:
            
            Nf = f.shape[0]
            Batch_size = x.shape[0]
            loss = 0
            for nf in range(Nf):
                fc = f[nf].unsqueeze(0).repeat(Batch_size,1,1,1)
                qf = -k**2*x*fc
                ipt = torch.cat((x, qf), dim=1)
                out = model(ipt)
                
                res = torch.sqrt(torch.sum((out-y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                ynorm = torch.sqrt(torch.sum((y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                loss += torch.sum(res/ynorm)/Nf

            train_l2 += loss.item()

    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:

            Nf = f.shape[0]
            Batch_size = x.shape[0]
            loss = 0
            for nf in range(Nf):
                fc = f[nf].unsqueeze(0).repeat(Batch_size,1,1,1)
                qf = -k**2*x*fc
                ipt = torch.cat((x, qf), dim=1)
                out = model(ipt)
                
                res = torch.sqrt(torch.sum((out-y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                ynorm = torch.sqrt(torch.sum((y[:,:,nf,:,:])**2, dim=(1, 2, 3)))
                loss += torch.sum(res/ynorm)/Nf

            test_l2 += loss.item()

    train_l2/= n_train
    test_l2 /= n_test
    
    return train_l2, test_l2