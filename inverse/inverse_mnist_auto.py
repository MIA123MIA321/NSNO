import torch
import torch.utils.data as Data
from torch import nn
from torch.nn import functional as F
import numpy as np
import scipy
from scipy.interpolate import interp2d
from datetime import datetime
from seaborn import heatmap
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from inverse import *
from models import *
import seaborn as sns

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

q_data = np.load('./mnist.npz')
q = q_data['x_test'][34]

q = q*1.0

N = q.shape[0]-1
N_buffer = int(N*14/128)
N_int = N-2*N_buffer
N_src = 32

k = 20

Q = q.reshape(-1,)
tmp_data = np.load('./bd_data34.npz')
f_original, partial_data_original = tmp_data['f'], tmp_data['p']

# f_real = torch.from_numpy(f.real).float()
# f_imag = torch.from_numpy(f.imag).float()
# f = torch.stack((f_real, f_imag), dim=1).to(device)

# partial_data_real = torch.from_numpy(partial_data.real).float()
# partial_data_imag = torch.from_numpy(partial_data.imag).float()
# partial_data = torch.stack((partial_data_real, partial_data_imag), dim=1).to(device)

print(q.shape)
print(f_original.shape)
print(partial_data_original.shape)

def gaussian_noise(x, noise_level):
    std = noise_level * np.std(x)
    noise = np.random.normal(0, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy

noise_level_data = 0
partial_data = gaussian_noise(partial_data_original, noise_level_data)
# np.save("noisy_data_q34_100", partial_data)

# partial_data = np.load("noisy_data_q34_10.npy")

noise_level_f = 0
f = np.zeros_like(f_original)
for i in range(f.shape[0]):
    f[i] = gaussian_noise(f_original[i], noise_level_f)
# np.save("noisy_f_30", f)

model = Neumann_MyUNO_mnist(N = 128, k = k, modes1 = 12, modes2 = 12, width = 32, FNO_act = 'gelu', CNN_act = 'gelu', num_layer = 3, DC_in = True).to(device)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

filename_model = 'UNO_nqf_mnist_128_dw1_pw01_1.pth'

model.load_state_dict(torch.load(filename_model))

# data_q0 = np.load('q34.npz')
# Q0 = data_q0['q_initial'].reshape(-1,)
Q0 = np.zeros_like(Q)

# X_list = []
# iters = 0
X_list.append(Q0)

regularization = None

args1 = (N, N_buffer, N_src, k, f, partial_data, model, device, True, "auto", regularization)
args2 = (N, N_buffer, N_src, k, f, partial_data, model, device, False, "auto", regularization)
J00 = J_single_frequency(Q0, *args2)
Jtt = J_single_frequency(Q, *args2)
print(J00)
print(Jtt)

J, DJ = J_single_frequency(Q, *args1)
print(J)
print(DJ.shape)

ftol = 1e-5 * J00

t0 = time.time()
_,Q0 = SOLVE(J_single_frequency, Q0=Q0, args=args1, jac=True,
        options={'disp': True,'gtol': 1e-6,
                 'maxiter': 20,'ftol':ftol},
        method='L-BFGS-B')
print('********************************************')
time_total = time.time() - t0
print(time_total)


ll = len(X_list)
# print(ll)
Error_list = []
q_record = np.zeros((ll, N+1, N+1))
for j in range(ll):
    Error_list.append(Error(X_list[j], Q))
    q_record[j] = X_list[j].reshape(N+1, N+1)

print(Error_list)

plt.figure(1, figsize=(4, 3), dpi=200)
h=sns.heatmap(q, cbar=False)
plt.xticks([], [])
plt.yticks([], [])
cb = h.figure.colorbar(h.collections[0])
cb.ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig('./figures/inv_qT38.png',bbox_inches='tight', pad_inches=0)

plt.figure(2, figsize=(4, 3), dpi=200)
h=sns.heatmap(q_record[ll-1], cbar=False)
plt.yticks([], [])
cb = h.figure.colorbar(h.collections[0])
cb.ax.tick_params(labelsize=15)
plt.tight_layout()
# plt.savefig('./figures/inv_nu.png',bbox_inches='tight', pad_inches=0)

print(np.sqrt(np.sum(DJ**2))/(N+1))

# np.savez('inverse_nu.npz', q=q, q_record=q_record, error = Error_list)