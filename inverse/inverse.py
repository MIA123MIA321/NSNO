import torch
import torch.utils.data as Data
from torch import nn
from torch.nn import functional as F
import numpy as np
import scipy
from scipy.interpolate import interp2d
from datetime import datetime
import time
from scipy.optimize import minimize

global grad_list
grad_list = []
X_list = []
iters = 0

# def data_projection(x, N, N_buffer, data_to_boundary = True):
#     # adjacent sides
#     # x.shape = (N+1)^2 / 4*(N_rec-2*N_buffer_rec-1)

#     index = np.arange(N_buffer+1, N-N_buffer)
#     if data_to_boundary:
#         x1 = x[index, N_buffer]
#         x2 = x[N_buffer, index]
#         x3 = x[index, -N_buffer-1]
#         x4 = x[-N_buffer-1, index]
#         return np.concatenate([x1, x2, x3, x4])
#     else:
#         N_int = N-2*N_buffer
#         output = np.zeros((N+1,N+1), dtype = np.complex128)
#         output[index, N_buffer] = x[0*(N_int-1):1*(N_int-1)]
#         output[N_buffer, index] = x[1*(N_int-1):2*(N_int-1)]
#         output[index, -N_buffer-1] = x[2*(N_int-1):3*(N_int-1)]
#         output[-N_buffer-1, index] = x[3*(N_int-1):4*(N_int-1)]
#         return output

# def Net_solve(Q, f, model, device, N, k, one_dim):
#     q = Q.reshape(N+1, N+1)
#     q = torch.from_numpy(q).float().unsqueeze(0).unsqueeze(0).to(device)
#     f = f.reshape(N+1, N+1)
#     f_real = torch.from_numpy(f.real).float()
#     f_imag = torch.from_numpy(f.imag).float()
#     f = torch.stack((f_real, f_imag), dim=0).unsqueeze(0).to(device)
#     ipt = torch.cat((q,f), dim=1)
    
#     u = model(ipt).squeeze(0)
#     u = (u[0,:,:]+1j*u[1,:,:]).cpu().detach().numpy()
#     if one_dim:
#         u = u.reshape(-1,)
#     return u
    
def J_single_frequency(Q, *argsargs):
    N, N_buffer, N_src, k, f, partial_data, model, device, return_grad, grad_type, regularization = argsargs
    if grad_type == "adjoint":
        return J_Net_adjoint(Q, N, N_buffer, N_src, k, f, partial_data, model, device, return_grad)
    elif grad_type == "auto":
        return J_Net_auto(Q, N, N_buffer, N_src, k, f, partial_data, model, device, return_grad, regularization)

# def J_Net(Q, N, N_buffer, N_src, k, f, partial_data, model, device, return_grad):
#     J_value = 0.
#     if return_grad:
#         J_grad = np.zeros_like(Q)
#     for j in range(N_src):
#         phi = Net_solve(Q, -k**2*Q*f[j].reshape(-1,), model, device, N, k, one_dim = False)
#         J_inner = data_projection(phi, N, N_buffer, True) - partial_data[j]
#         J_value += np.linalg.norm(J_inner, ord = 2)**2
#         if return_grad:
#             fun1 = (f[j] + phi).reshape(-1,)
#             fun2 = - k**2 * data_projection(J_inner, N, N_buffer, False).reshape(-1,)
            
#             tmp_fun = Net_solve(Q, fun2.real, model, device, N, k, one_dim = True)
#             tmpr = fun1.real * tmp_fun.real - fun1.imag * tmp_fun.imag
#             tmp_fun = Net_solve(Q, fun2.imag, model, device, N, k, one_dim = True)
#             tmpi = fun1.imag * tmp_fun.real + fun1.real * tmp_fun.imag
            
#             # tmp_fun = Net_solve(Q, fun2.real+1j*fun2.real, model, device, N, k, one_dim = True)
#             # tmpr = fun1.real * (tmp_fun.real + tmp_fun.imag)/2 - fun1.imag * (tmp_fun.imag - tmp_fun.real)/2
#             # tmp_fun = Net_solve(Q, fun2.imag+1j*fun2.imag, model, device, N, k, one_dim = True)
#             # tmpi = fun1.imag * (tmp_fun.real + tmp_fun.imag)/2 + fun1.real * (tmp_fun.imag - tmp_fun.real)/2
            
#             J_grad += (tmpr + tmpi)
#     J_value = 0.5 * J_value / N_src
#     if return_grad:
#         J_grad = J_grad / N_src
#         print(J_value)
#         print(J_grad)
#         return J_value, J_grad
#     else:
#         return J_value

def data_projection(x, N, N_buffer, device, data_to_boundary = True):
    # adjacent sides
    # x.shape = (N+1)^2 / 4*(N_rec-2*N_buffer_rec-1)

    index = torch.arange(N_buffer+1, N-N_buffer)
    if data_to_boundary:
        x1 = x[:, index, N_buffer]
        x2 = x[:, N_buffer, index]
        x3 = x[:, index, -N_buffer-1]
        x4 = x[:, -N_buffer-1, index]
        return torch.cat([x1, x2, x3, x4], dim=1)
    else:
        N_int = N-2*N_buffer
        N_src = x.shape[0]
        output = torch.zeros(N_src, N+1, N+1, dtype = torch.cfloat).to(device)
        output[:, index, N_buffer] = x[:, 0*(N_int-1):1*(N_int-1)]
        output[:, N_buffer, index] = x[:, 1*(N_int-1):2*(N_int-1)]
        output[:, index, -N_buffer-1] = x[:, 2*(N_int-1):3*(N_int-1)]
        output[:, -N_buffer-1, index] = x[:, 3*(N_int-1):4*(N_int-1)]
        return output
    
# def J_Net(Q, N, N_buffer, N_src, k, f, partial_data, model, device, return_grad):
#     J_value = 0.
#     if return_grad:
#         J_grad = np.zeros_like(Q)
#     q_torch = torch.from_numpy(Q.reshape(N+1, N+1)).float().unsqueeze(0).unsqueeze(0).repeat(N_src, 1, 1, 1).to(device)
#     f_real = torch.from_numpy(f.real).float()
#     f_imag = torch.from_numpy(f.imag).float()
#     f_torch = torch.stack((f_real, f_imag), dim=1).to(device)
#     q_torch1 = q_torch.repeat(1, 2, 1, 1)
#     ipt = torch.cat((q_torch, -k**2*q_torch1*f_torch), dim=1)
#     phi = model(ipt)
#     phi = phi[:,0,:,:] + 1j * phi[:,1,:,:]
#     J_inner = data_projection(phi, N, N_buffer, device, True) - torch.from_numpy(partial_data).cfloat().to(device)
#     J_value = 0.5*torch.norm(J_inner).item() ** 2 / N_src
    
#     if return_grad:
#         fun1 = f_torch[:,0,:,:] + 1j*f_torch[:,1,:,:] + phi
#         fun2 = - k**2 * data_projection(J_inner, N, N_buffer, device, False)
        
#         ipt1 = torch.stack((q_torch.squeeze(1), fun2.real, torch.zeros_like(fun2.real)), dim = 1)
#         tmp_fun1 = model(ipt1)
#         tmpr = fun1.real * tmp_fun1[:,0,:,:] - fun1.imag * tmp_fun1[:,1,:,:]
        
#         ipt2 = torch.stack((q_torch.squeeze(1), fun2.imag, torch.zeros_like(fun2.imag)), dim = 1)
#         tmp_fun2 = model(ipt2)
#         tmpi = fun1.imag * tmp_fun2[:,0,:,:] + fun1.real * tmp_fun2[:,1,:,:]
        
#         J_grad = torch.sum(tmpr + tmpi, dim = 0) / N_src
#         J_grad = J_grad.reshape(-1,).cpu().detach().numpy()
        
#         J_grad = J_grad.astype(np.float64)
        
#         return J_value, J_grad
#     else:
#         return J_value

# adjoint
def J_Net_adjoint(Q, N, N_buffer, N_src, k, f, partial_data, model, device, return_grad):
    J_value = 0.
    if return_grad:
        J_grad = np.zeros_like(Q)
    q_torch = torch.from_numpy(Q.reshape(N+1, N+1)).float().unsqueeze(0).unsqueeze(0).repeat(N_src, 1, 1, 1).to(device)
    f_real = torch.from_numpy(f.real).float()
    f_imag = torch.from_numpy(f.imag).float()
    f_torch = torch.stack((f_real, f_imag), dim=1).to(device)
    q_torch1 = q_torch.repeat(1, 2, 1, 1)
    ipt = torch.cat((q_torch, -k**2*q_torch1*f_torch), dim=1)
    phi = model(ipt)
    phi = phi[:,0,:,:] + 1j * phi[:,1,:,:]
    J_inner = data_projection(phi, N, N_buffer, device, True) - torch.from_numpy(partial_data).cfloat().to(device)
    J_value = 0.5*torch.norm(J_inner).item() ** 2 / N_src
    
    if return_grad:
        fun1 = f_torch[:,0,:,:] + 1j*f_torch[:,1,:,:] + phi
        fun2 = - k**2 * data_projection(J_inner, N, N_buffer, device, False)
        
        ipt1 = torch.stack((q_torch.squeeze(1), fun2.real, -fun2.imag), dim = 1)
        tmp_fun1 = model(ipt1)
        tmp = fun1.real * tmp_fun1[:,0,:,:] - fun1.imag * tmp_fun1[:,1,:,:]
        
        J_grad = torch.sum(tmp, dim = 0) / N_src
        J_grad = J_grad.reshape(-1,).cpu().detach().numpy()
        
        J_grad = J_grad.astype(np.float64)

        grad_list.append(J_grad)
        # print(np.max(np.abs(J_grad)))
        
        return J_value, J_grad
    else:
        return J_value

# auto    
def J_Net_auto(Q, N, N_buffer, N_src, k, f, partial_data, model, device, return_grad, regularization):
    J_value = 0.
    if return_grad:
        J_grad = np.zeros_like(Q)
    q_torch = torch.from_numpy(Q.reshape(N+1, N+1)).float().to(device)
    q_torch.requires_grad = True
    q_torch1 = q_torch.unsqueeze(0).unsqueeze(0).repeat(N_src, 1, 1, 1)
    f_real = torch.from_numpy(f.real).float()
    f_imag = torch.from_numpy(f.imag).float()
    f_torch = torch.stack((f_real, f_imag), dim=1).to(device)
    q_torch2 = q_torch1.repeat(1, 2, 1, 1)
    ipt = torch.cat((q_torch1, -k**2*q_torch2*f_torch), dim=1)
    phi = model(ipt)
    phi = phi[:,0,:,:] + 1j * phi[:,1,:,:]
    J_inner = data_projection(phi, N, N_buffer, device, True) - torch.from_numpy(partial_data).cfloat().to(device)
    J_value = 0.5*torch.norm(J_inner) ** 2 / N_src
    

    if regularization == "TV":
        Q_TV = torch.sum(torch.abs(q_torch[:, 1:]-q_torch[:,:-1]))+torch.sum(torch.abs(q_torch[1:,:]-q_torch[:-1,:]))
        J_value = J_value + 0.0001*Q_TV
    elif regularization == "H1":
        Q_H1 = torch.sum(torch.abs(q_torch[:, 1:]-q_torch[:,:-1])**2)+torch.sum(torch.abs(q_torch[1:,:]-q_torch[:-1,:])**2)
        J_value = J_value + 0.01*Q_H1
    
    if return_grad:
        J_value.backward()
        
        J_grad1 = q_torch.grad
        J_grad = J_grad1

        J_grad = J_grad[1::2, 1::2].unsqueeze(0).unsqueeze(0)
        J_grad = F.interpolate(J_grad, size=(N-1, N-1), mode='bilinear', align_corners=True)
        J_grad = F.pad(J_grad, (1, 1, 1, 1), "replicate")

        J_grad = J_grad.reshape(-1,).cpu().detach().numpy()
        J_grad = J_grad.astype(np.float64)
        
        J_value = J_value.item()

        grad_list.append(J_grad)
        # print(np.max(np.abs(J_grad)))
        
        return J_value, J_grad
    else:
        J_value = J_value.item()
        return J_value
    
    
def SOLVE(fun, Q0, args, jac, method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-5, 'maxiter': 100}):
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(fun, x0 = Q0, args = args, method = method, jac = jac, bounds = [(0, None) for i in range(129**2)],
                        options = options, callback = callbackF)
        return res.success, res.x

def callbackF(X):
    global X_list
    global iters
    X_list.append(X)
    iters += 1
    if iters >= 1 and iters < 99:
        print('iter {} completed'.format(iters),
          '       %s' % str(datetime.now())[:-7])
    else:
        print('iter {}  completed'.format(iters),
              '       %s' % str(datetime.now())[:-7])
        
def Error(a, a_truth):
    tmp = np.linalg.norm((a - a_truth), ord=2)
    return tmp / np.linalg.norm(a_truth, ord=2)

def get_grad_record(x = True):
    return grad_list

# def NORM1(g,N=129):
#     tmp_g = g.reshape(N,N)
#     laplace_g = 0.25*(tmp_g[2:,1:-1]+tmp_g[:-2,1:-1]+tmp_g[1:-1,2:]+tmp_g[1:-1,:-2]) - tmp_g[1:-1,1:-1]
#     sorted_indices = list(np.argsort(np.abs(laplace_g.reshape(-1,))))
#     sorted_indices1 = [(_//(N-2)+1,_%(N-2)+1) for _ in sorted_indices]
#     for i in range(3):
#         id1, id2 = sorted_indices1[-1-i]
#         id0 = id1*N+id2
#         g[id0] = 0.25*(g[id0-1]+g[id0+1]+g[id0+N]+g[id0-N])
#         # g[id0] = 0.4*(g[id0-1]+g[id0+1]+g[id0+N]+g[id0-N])-0.1*(g[id0-1-N]+g[id0+1+N]+g[id0+N-1]+g[id0-N+1])-0.05*(g[id0-2]+g[id0+2]+g[id0+N+N]+g[id0-N-N])
#     return g

# def NORM2(g,N=129):
#     tmp_g = g.reshape(N,N)

#     sorted_indices = list(np.argsort(np.abs(g)))[::-1]
#     tmp = g[sorted_indices[0]]
#     i = 0
#     # while np.abs(g[sorted_indices[i]]) * 0.8 < np.abs(g[sorted_indices[i+1]]) and np.abs(g[sorted_indices[i]]) * 0.8 > np.abs(g[sorted_indices[i+2]]):
#     while ((np.abs(g[sorted_indices[i]])-np.abs(g[sorted_indices[i+1]]))/2>np.abs(g[sorted_indices[i+1]])- np.abs(g[sorted_indices[i+2]]) and np.abs(g[sorted_indices[i]]) * 0.8 > np.abs(g[sorted_indices[i+1]])) or np.abs(g[sorted_indices[i]]) * 0.8 < np.abs(g[sorted_indices[i+1]]) and np.abs(g[sorted_indices[i]]) * 0.8 > np.abs(g[sorted_indices[i+2]]):
#         g[sorted_indices[i]] = 0.25*(g[sorted_indices[i]+1] + g[sorted_indices[i]-1]+g[sorted_indices[i]+N] + g[sorted_indices[i]-N])
#         i += 1
#     return g


def NORM3(g, N=129):
    def NORM_fun(g, N=129):
        tmp_g = g.reshape(N, N)
        laplace_g = 0.25*(tmp_g[2:, 1:-1]+tmp_g[:-2, 1:-1] +
                          tmp_g[1:-1, 2:]+tmp_g[1:-1, :-2]) - tmp_g[1:-1, 1:-1]

        sorted_indice = list(np.argsort(np.abs(laplace_g.reshape(-1,))))[-1]
        id1, id2 = (sorted_indice//(N-2)+1, sorted_indice % (N-2)+1)
        id0 = id1*N+id2
        # g[id0] = 0.25*(g[id0-1]+g[id0+1]+g[id0+N]+g[id0-N])
        try:
            g[id0] = 0.4*(g[id0-1]+g[id0+1]+g[id0+N]+g[id0-N])-0.1*(g[id0-1-N]+g[id0+1+N] +
                                                                    g[id0+N-1]+g[id0-N+1])-0.05*(g[id0-2]+g[id0+2]+g[id0+N+N]+g[id0-N-N])
        except:
            g[id0] = 0.25*(g[id0-1]+g[id0+1]+g[id0+N]+g[id0-N])
        return g, laplace_g.max(), laplace_g.min()
    g, MAX, MIN = NORM_fun(g, N)
    i = 0
    while i <= 10:
        g_tmp, MAX_tmp, MIN_tmp = NORM_fun(g, N)
        if abs(MAX_tmp) <= abs(MAX)*0.8 or abs(MIN_tmp) <= abs(MIN)*0.8:
            g, MAX, MIN = (g_tmp, MAX_tmp, MIN_tmp)
            i += 1
        else:
            g_tmp1, MAX_tmp1, MIN_tmp1 = NORM_fun(g_tmp, N)
            if abs(MAX_tmp1) <= abs(MAX)*0.8 or abs(MIN_tmp1) <= abs(MIN)*0.8:
                g, MAX, MIN = (g_tmp1, MAX_tmp1, MIN_tmp1)
                i += 2
            else:
                break
    # for i in range(5):
    #     g,MAX,MIN = NORM_fun(g,N)

    return g