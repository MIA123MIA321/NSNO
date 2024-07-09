# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:00:51 2023

@author: c
"""

from scipy.sparse.linalg import cg
from mumps import DMumpsContext
import random
import numpy as np
import scipy
from scipy.sparse import dia_matrix
import time
from datetime import datetime
from seaborn import heatmap
import matplotlib.pyplot as plt
import time
from scipy.fftpack import idct

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ctx = DMumpsContext()
ctx.set_silent()

def gen_coef_gauss9(N_sample, al, au, gl, gu):
    coef = np.zeros((N_sample, 9, 3))
    for n in range(N_sample):
        coef[n,0:9,0] = np.random.rand(9)*(au-al)+al
        coef[n,0:9,1] = np.random.rand(9)*(au-al)+al
        coef[n,0:9,2] = np.random.rand(9)*(gu-gl)+gl
    return coef

def gen_coef_tri4(N_sample):
    coef = np.zeros((N_sample, 6, 2))
    for n in range(N_sample):
        for nn in range(6):
            al = 2**nn
            au = 1.5*2**nn
            coef[n, nn, 0] = np.random.rand(1)*(au-al)+al
            coef[n, nn, 1] = np.random.rand(1)*2*np.pi
    return coef

def gen_coef_t(N_sample, N, xl = 0.05, xu = 0.95, yl = 0.05, yu = 0.95):
    coef = np.zeros((N_sample, 8), dtype = np.int32)
    for n in range(N_sample):
        x_axis = [int(random.uniform(xl, xu)*N) for i in range(3)]
        x_axis.sort()
        y_axis = [int(random.uniform(yl, yu)*N) for i in range(4)]
        y_axis.sort()
        coef[n, 0:3] = x_axis
        coef[n, 3:7] = y_axis
        coef[n, 7] = random.randint(0,3)
    return coef

def gen_coef_C(N_sample, N, xl, xu, rl, ru, gl, gu):
    coef = np.zeros((N_sample, 3, 4))
    for n in range(N_sample):
        Ng = np.random.randint(1, 4)
        coef[n, 0:Ng, 0] = np.random.rand(Ng)*(xu-xl)+xl
        coef[n, 0:Ng, 1] = np.random.rand(Ng)*(xu-xl)+xl
        coef[n, 0:Ng, 2] = np.random.rand(Ng)*(ru-rl)+rl
        coef[n, 0:Ng, 3] = np.random.rand(Ng)*(gu-gl)+gl
    return coef

def f_gen_Gauss9(N, coef):
    N_sample = coef.shape[0]
    f = np.zeros((N_sample, N + 1, N + 1))
    for n in range(N_sample):
        for ii in range(3):
            for jj in range(3):
                x0 = 0.3*ii+0.2
                y0 = 0.3*jj+0.2
                ind = 3*ii+jj
                f[n,:,:] += gen_Gauss(N, coef[n,ind,0], x0, coef[n,ind,1], y0, coef[n,ind,2])
        max_f = np.max(np.abs(f[n,:,:]))
        f[n,:,:] = f[n,:,:]/max_f
    return f

def f_gen_tri4(N, coef):
    N_sample = coef.shape[0]
    f = np.zeros((N_sample, N + 1, N + 1))
    tmp = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(tmp, tmp)
    for n in range(N_sample):
        for nn in range(6):
            f[n,:,:] += 1/coef[n, nn, 0]*np.cos(coef[n, nn, 0]*np.pi*(X*np.cos(coef[n, nn, 1])+Y*np.sin(coef[n, nn, 1])))
        max_f = np.max(np.abs(f[n,:,:]))
        f[n,:,:] = f[n,:,:]/max_f
    return f

def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def f_gen_GRF(N, N_sample, alpha, tau):
    f = np.zeros((N_sample, N+1, N+1))
    for n in range(N_sample):
        tmp = np.linspace(0, 1, N+1)
        X, Y = np.meshgrid(tmp, tmp)
        xi = np.random.randn(N+1, N+1)
        K1, K2 = np.meshgrid(np.arange(N+1), np.arange(N+1))
        coef = tau**(alpha-1)*(np.pi**2*(K1**2+K2**2) + tau**2)**(-alpha/2)
        L = (N+1)*coef*xi
        L[0, 0] = 0
        f[n, :, :] = idct2(L)
        max_f = np.max(np.abs(f[n, :, :]))
        f[n, :, :] = f[n, :, :]/max_f
    return f

def q_gen_t(N, coef, value1, value2):
    N_sample = coef.shape[0]
    q = np.zeros((N_sample, N+1, N+1))
    for n in range(N_sample):
        x_axis = coef[n, 0:3]
        y_axis = coef[n, 3:7]
        direct = coef[n, 7]
        qq = np.zeros((N+1, N+1))
        qq[x_axis[0]:x_axis[1], y_axis[0]:y_axis[3]] = value1
        qq[x_axis[1]:x_axis[2], y_axis[1]:y_axis[2]] = value2
        if direct==1:
            q[n,:,:] = qq[::-1,:]
        elif direct==2:
            q[n,:,:] = qq.T
        elif direct==3:
            q[n,:,:] = qq[::-1,:].T
        else:
            q[n,:,:] = qq
    return q

def q_gen_Gauss4(N, coef, value1):
    N_sample = coef.shape[0]
    q = np.zeros((N_sample, N + 1, N + 1))
    for n in range(N_sample):
        for ii in range(2):
            for jj in range(2):
                x0 = 0.5*ii+0.25
                y0 = 0.5*jj+0.25
                ind = 2*ii+jj
                q[n,:,:] += gen_Gauss(N, coef[n,ind,0], x0, coef[n,ind,1], y0, coef[n,ind,2])
        max_q = np.max(np.abs(q[n,:,:]))
        q[n,:,:] = q[n,:,:]/max_q*value1
    return q

def q_gen_C(N, coef, value1):
    N_sample = coef.shape[0]
    q = np.zeros((N_sample, N + 1, N + 1))
    tmp = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(tmp, tmp)
    for n in range(N_sample):
        for ng in range(3):
            a = coef[n, ng, 0]
            b = coef[n, ng, 1]
            r = coef[n, ng, 2]
            g = coef[n, ng, 3]
            q[n,:,:] += g*((X-a)**2+(Y-b)**2<=r**2).astype(np.float64)
        max_q = np.max(np.abs(q[n,:,:]))
        q[n,:,:] = q[n,:,:]/max_q*value1
    return q

def q_gen_C2(N, coef, value1):
    N_sample = coef.shape[0]
    q = np.zeros((N_sample, N + 1, N + 1))
    tmp = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(tmp, tmp)
    for n in range(N_sample):
        for ng in range(3):
            a = coef[n, ng, 0]
            b = coef[n, ng, 1]
            r = coef[n, ng, 2]
            g = coef[n, ng, 3]
            if g != 0:
                dis = 1-((X-a)**2+(Y-b)**2)/r**2
                # dis = np.maximum(dis, 0)
                circle = g*((X-a)**2+(Y-b)**2<=r**2).astype(np.float64)*np.exp(-1/dis)
                q[n,:,:] += np.nan_to_num(circle)
        max_q = np.max(np.abs(q[n,:,:]))
        q[n,:,:] = q[n,:,:]/max_q*value1
    return q

def Matrix_Gen(N, Q, k):
    M = N + 1
    data1 = k * k * (1 + Q) - 4 * N * N
    data1 = np.tile(data1, 2)
    data2 = np.ones(M).reshape(-1, )
    data2[0] = 0
    data2[1] = 2
    data2 = np.tile(data2, 2)
    data2_plus = N * N * np.tile(data2, M)
    data2_minus = np.flipud(data2_plus)
    data3 = np.ones(M * M).reshape(-1, )
    data3[:M] = 0
    data3[M:2 * M] = 2
    data3_plus = N * N * data3
    data3_plus = np.tile(data3_plus, 2)
    data3_minus = np.flipud(data3_plus)
    matrix__ = np.ones((M, M))
    matrix__[0, 0] = matrix__[-1, 0] = matrix__[-1, -1] = matrix__[0, -1] = 2
    matrix__[1:-1, 1:-1] = 0
    data4 = -2 * k * matrix__.reshape(-1, ) * N
    data4_plus = np.tile(data4, 2)
    data4_minus = -data4_plus
    data = (
        np.c_[
            data1,
            data2_minus,
            data2_plus,
            data3_minus,
            data3_plus,
            data4_minus,
            data4_plus]).transpose()
    offsets = np.array([0, -1, 1, -M, M, -M * M, M * M])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    return dia.tocoo()

def Matrix_analysis(N: int, Q: np.ndarray, k: float) -> None:
    """
    Q.shape = ((N+1)**2,1)
    """
    global ctx
    _Matrix_ = Matrix_Gen(N, Q, k)
    ctx.set_shape(_Matrix_.shape[0])
    if ctx.myid == 0:
        ctx.set_centralized_assembled_rows_cols(
            _Matrix_.row + 1, _Matrix_.col + 1)
    ctx.run(job=1)

def Matrix_factorize(N: int, Q: np.ndarray, k: float) -> None:
    global ctx
    _Matrix_ = Matrix_Gen(N, Q, k)
    ctx.set_centralized_assembled_values(_Matrix_.data)
    ctx.run(job=2)
    return

def Matrix_solve(F: np.ndarray) -> np.ndarray:
    """
    Q.shape = ((N+1)**2,1)
    F.shape = ((N+1)**2,1)
    Returns.shape = ((N+1)**2,)
    """
    global ctx
    M = F.shape[0]
    F = np.append(F.real,F.imag)
    _Right_ = F
    x = _Right_.copy()
    ctx.set_rhs(x)
    ctx.run(job=3)
    tmp = x.reshape(-1, )
    return tmp[:M]+1j*tmp[M:]

def mathscr_F0(
        N: int,
        Q: np.ndarray,
        k: float,
        F: np.ndarray,
        solver: str = 'CG') -> np.ndarray:
    """
    Q.shape = ((N+1)**2,)
    F.shape = ((N+1)**2,)
    Returns.shape = ((N+1)**2,)
    """
    global ctx
    # if solver == 'CG':
    #     return CG_method(N, Q, k, F)
    if solver == 'MUMPS':
        Matrix_factorize(N, Q, k)
        return Matrix_solve(F)

k = 20
f_method = 'GRF'
q_method = 't'
s = 256
comp_s = 256
times = comp_s//s
N_sample = 100
maxq = 0.1
alpha = 2
tau = 3

elif f_method == 'Gauss9':
    coef_f = gen_coef_gauss9(N_sample, al=30, au=60, gl=-1, gu=1)
    f = k**2*f_gen_Gauss9(comp_s, coef_f)
elif f_method == 'GRF':
    f = k**2*f_gen_GRF(comp_s, N_sample, alpha = alpha, tau = tau)
elif f_method == 'tri4':
    coef_f = gen_coef_tri4(N_sample)
    f = k**2*f_gen_tri4(comp_s, coef_f)

if q_method == 't':
    coef_q = gen_coef_t(N_sample, comp_s, xl = 0.05, xu = 0.95, yl = 0.05, yu = 0.95)
    q = q_gen_t(comp_s, coef_q, value1 = maxq, value2 = maxq)
elif q_method == 'C':
    coef_q = gen_coef_C(N_sample, comp_s, xl=0.2, xu=0.8, rl=0.05, ru=0.2, gl=-1, gu=1)
    q = q_gen_C(comp_s, coef_q, value1 = maxq)
elif q_method == 'C2':
    coef_q = gen_coef_C(N_sample, comp_s, xl=0.2, xu=0.8, rl=0.05, ru=0.2, gl=-1, gu=1)
    q = q_gen_C2(comp_s, coef_q, value1 = maxq)

print('Generation of q and f completed')

exact = np.zeros((N_sample, s+1, s+1), dtype=complex)
# ini = np.zeros((N_sample, s+1, s+1), dtype=complex)

for n in range(N_sample):
    t1 = time.time()
    F = f[n,:,:].reshape(-1,)
    Q = q[n,:,:].reshape(-1,)
    
    Matrix_analysis(comp_s, Q, k)
    exact_comp = mathscr_F0(comp_s, Q, k, F, 'MUMPS')
    exact[n,:,:] = exact_comp.reshape((comp_s+1, comp_s+1))[::times,::times]

    # ini_comp = mathscr_F0(comp_s, Q*0, k, F, 'MUMPS')
    # ini[n,:,:] = ini_comp.reshape((comp_s+1, comp_s+1))[::times,::times]
    
    t2 = time.time()
    print(n)
    print(t2-t1)
    
ctx.destroy()

# f = f[:,::times,::times]
q = q[:,::times,::times]

filename_data = 'ftest_qT_01_fGRF_23_k20.npz'

if f_method == 'GRF':
    np.savez(filename_data, N_sample=N_sample, k=k, s=s, times=times, comp_s=comp_s,
              f=f, alpha=alpha, tau=tau, coef_q=coef_q, q=q, exact=exact, maxq=maxq)
else:
    np.savez(filename_data, N_sample=N_sample, k=k, s=s, times=times, comp_s=comp_s,
          coef_f=coef_f, f=f, coef_q=coef_q, q=q, exact=exact, maxq=maxq)
