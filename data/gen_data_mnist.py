# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:31:57 2023

@author: c
"""

# from scipy.sparse.linalg import cg
from mumps import DMumpsContext
# import random
import numpy as np
import scipy
from scipy.interpolate import interp2d
# from datetime import datetime
# from seaborn import heatmap
import matplotlib.pyplot as plt
import time
from Solver import *

q_data = np.load('mnist.npz')
q_train = q_data['x_train']
q_test = q_data['x_test']
N_train = q_train.shape[0]
N_test = q_test.shape[0]

def f_gen(N, k, m):
    res = np.zeros((m, (N+1), (N+1)), dtype = np.complex128)
    tmp = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(tmp, tmp)
    for j in range(m):  
        res[j] = np.exp(1j*k*(X*np.cos(2*np.pi*j/M)+Y*np.sin(2*np.pi*j/m)))
    return res

def INTERPOLATE(x, in_size, out_size):
    l_in = np.linspace(0, 1, in_size + 1)
    l_out = np.linspace(0, 1, out_size + 1)
    output = interp2d(l_in, l_in, x, kind='cubic')(l_out, l_out)
    return output

N_gen = 256
N = 128
k = 20
m = 4
times = 2

f = f_gen(N_gen, k, m) # m*(N_gen+1)*(N_gen+1)

exact_train = np.zeros((N_train, m, N+1, N+1), dtype=complex)
exact_test = np.zeros((N_test, m, N+1, N+1), dtype=complex)

Q_train = np.zeros((N_train, N+1, N+1), dtype=float)
Q_test = np.zeros((N_test, N+1, N+1), dtype=float)

for n in range(N_train):
    t1 = time.time()

    coef = 1
    Q = INTERPOLATE(q_train[n,:,:]*coef, N, N_gen)
    Q_train[n] = Q[::2,::2]
    Q = Q.reshape(-1,)

    Matrix_analysis(N_gen, Q, k)
    Matrix_factorize(N_gen, Q, k)
    for mm in range(m):
        u_tmp = Matrix_solve(-k**2*Q*f[mm].reshape(-1,), False)
        exact_train[n, mm, :, :] = u_tmp[::times,::times]
        
    t2 = time.time()
    print(n)
    print(t2-t1)

for n in range(N_test):
    t1 = time.time()

    coef = 1
    Q = INTERPOLATE(q_test[n,:,:]*coef, N, N_gen)
    Q_test[n] = Q[::2,::2]
    Q = Q.reshape(-1,)

    Matrix_analysis(N_gen, Q, k)
    Matrix_factorize(N_gen, Q, k)
    for mm in range(m):
        u_tmp = Matrix_solve(-k**2*Q*f[mm].reshape(-1,), False)
        exact_test[n, mm, :, :] = u_tmp[::times,::times]
        
    t2 = time.time()
    print(n)
    print(t2-t1)

ctx.destroy()

f = f[:,::times,::times]

filename_data = 'mnist_train.npz'
np.savez(filename_data, exact_train=exact_train, exact_test=exact_test, q_train=Q_train, q_test=Q_test, f=f)