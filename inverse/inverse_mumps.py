# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 03:31:14 2023

@author: c
"""

from mumps import DMumpsContext
# import random
import numpy as np
import scipy
from scipy.interpolate import interp2d
from datetime import datetime
# from seaborn import heatmap
import matplotlib.pyplot as plt
import time
from Solver import *
from scipy.optimize import minimize

q_data = np.load('mnist.npz')
q = q_data['x_test'][34]

q = q*1.0

def INTERPOLATE(x, in_size, out_size):
    l_in = np.linspace(0, 1, in_size + 1)
    l_out = np.linspace(0, 1, out_size + 1)
    output = interp2d(l_in, l_in, x, kind='cubic')(l_out, l_out)
    return output

# plt.figure(1, dpi=300)
# plt.imshow(q, cmap='Greys')
# plt.colorbar()

def f_gen(N, k, m):
    res = np.zeros((m, (N+1), (N+1)), dtype = np.complex128)
    tmp = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(tmp, tmp)    
    for j in range(m):    
        res[j] = np.exp(1j*k*(X*np.cos(2*np.pi*j/m)+Y*np.sin(2*np.pi*j/m)))
    return res

def data_gen(Q, N, N_buffer, N_src, k):

    f = f_gen(N, k, N_src)
    Matrix_analysis(N, Q, k)
    Matrix_factorize(N, Q, k)

    partial_data = np.zeros((N_src, 4*(N-2*N_buffer-1)), dtype = np.complex128)
    for i in range(N_src):
        u = Matrix_solve(-k**2*Q*f[i].reshape(-1,), False)
        partial_data[i] = data_projection(u, N, N_buffer, True)
    return f, partial_data

def data_projection(x, N, N_buffer, data_to_boundary = True):
    # adjacent sides
    # x.shape = (N+1)^2 / 4*(N_rec-2*N_buffer_rec-1)

    index = np.arange(N_buffer+1, N-N_buffer)
    if data_to_boundary:
        x1 = x[index, N_buffer]
        x2 = x[N_buffer, index]
        x3 = x[index, -N_buffer-1]
        x4 = x[-N_buffer-1, index]
        return np.concatenate([x1, x2, x3, x4])
    else:
        N_int = N-2*N_buffer
        output = np.zeros((N+1,N+1), dtype = np.complex128)
        output[index, N_buffer] = x[0*(N_int-1):1*(N_int-1)]
        output[N_buffer, index] = x[1*(N_int-1):2*(N_int-1)]
        output[index, -N_buffer-1] = x[2*(N_int-1):3*(N_int-1)]
        output[-N_buffer-1, index] = x[3*(N_int-1):4*(N_int-1)]
        return output

N = q.shape[0]-1
N_buffer = int(N*14/128)
N_int = N-2*N_buffer
N_src = 32

k = 20

Q = q.reshape(-1,)
load_boundary = 'F'
if load_boundary == 'T':
    tmp_data = np.load('bd_data34.npz')
    f, partial_data = tmp_data['f'], tmp_data['p']
else:
    f, partial_data = data_gen(Q, N, N_buffer, N_src, k)
    np.savez('bd_data34.npz', f = f, p = partial_data)

print(f.shape)
print(partial_data.shape)

Q0 = np.zeros_like(Q)
X_list = []
iters = 0
X_list.append(Q0)
Matrix_analysis(N, Q0, k)

def J_single_frequency(Q, *argsargs):
    N, N_buffer, N_src, k, f, partial_data, return_grad = argsargs
    return J_MUMPS(Q, N, N_buffer, N_src, k, f, partial_data, return_grad)

def J_MUMPS(Q, N, N_buffer, N_src, k, f, partial_data, return_grad):
    J_value = 0.
    if return_grad:
        J_grad = np.zeros_like(Q)
    Matrix_factorize(N, Q, k)
    for j in range(N_src):
        phi = Matrix_solve(-k**2*Q*f[j].reshape(-1,), False)
        J_inner = data_projection(phi, N, N_buffer, True) - partial_data[j]
        J_value += np.linalg.norm(J_inner, ord = 2)**2
        if return_grad:
            fun1 = (f[j] + phi).reshape(-1,)
            fun2 = - k**2 * data_projection(J_inner, N, N_buffer, False).reshape(-1,)
            tmp_fun = Matrix_solve(fun2.real, True)
            tmpr = fun1.real * tmp_fun.real - fun1.imag * tmp_fun.imag
            tmp_fun = Matrix_solve(fun2.imag, True)
            tmpi = fun1.imag * tmp_fun.real + fun1.real * tmp_fun.imag
            J_grad += (tmpr + tmpi) 
    J_value = 0.5 * J_value / N_src
    if return_grad:
        J_grad = J_grad / N_src
        return J_value, J_grad
    else:
        return J_value

args1 = (N, N_buffer, N_src, k, f, partial_data, True)
args2 = (N, N_buffer, N_src, k, f, partial_data, False)
J00 = J_single_frequency(Q0, *args2)
Jtt = J_single_frequency(Q, *args2)
print(J00)
print(Jtt)

J, DJ = J_single_frequency(Q, *args1)
print(J)
print(DJ.shape)
print(np.sqrt(np.sum(DJ**2))/(N+1))

ftol = 1e-6 * J00

def SOLVE(fun, Q0, args, jac, method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-6, 'maxiter': 30}):
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(fun, x0 = Q0, args = args, method = method, jac = jac,
                       bounds = [(0, None)], options = options, callback = callbackF)
        return res.success, res.x

def callbackF(X):
    global X_list
    global iters
    X_list.append(X)
    iters += 1
    if iters >= 10 and iters < 99:
        print('iter {} completed'.format(iters),
          '       %s' % str(datetime.now())[:-7])
    else:
        print('iter {}  completed'.format(iters),
              '       %s' % str(datetime.now())[:-7])

t0 = time.time()
_,Q0 = SOLVE(J_single_frequency, Q0=Q0, args=args1, jac=True,
        options={'disp': True,'gtol': 1e-10,
                 'maxiter': 100,'ftol':ftol},
        method='L-BFGS-B')
print('********************************************')
time_total = time.time() - t0
print(time_total)

def Error(a, a_truth):
    tmp = np.linalg.norm((a - a_truth), ord=2)
    return tmp / np.linalg.norm(a_truth, ord=2)

ll = len(X_list)
Error_list = []
q_record = np.zeros((ll, N+1, N+1))
for j in range(ll):
    Error_list.append(Error(X_list[j], Q))
    q_record[j] = X_list[j].reshape(N+1, N+1)

print(Error_list)

ctx.destroy()

# np.savez('inverse_mumps_q34_03_k40.npz', q=q, q_record=q_record, error = Error_list)
