import torch
import torch.utils.data as Data
from torch import nn
from torch.nn import functional as F
# import matplotlib.pyplot as plt
import time
import sys
import scipy
from utilities import *
from models import *
from timeit import default_timer
import numpy as np
from UNet import *

from Adam import Adam

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED_SET(1)

operator = 'nqf' # f, qf, nqf
# if operator == 'f':
#     neumann_test = True

model_type = 'UNO' # FNO, UNO

DC_in = True
FNO_act = 'gelu'
CNN_act = 'gelu'

times = 1

data_weight = 1
PINN_weight = 0.1

num_layer = 3

data_name = 'mnist'

save_model = True
if save_model:
    filename_model = model_type + '_' + operator + '_' + data_name + '_128_dw1_pw01_1.pth'
    filename_data = model_type + '_' + operator + '_' + data_name + '_128_dw1_pw01_1.npz'

# load data
N_train, N_test, q_train, q_test, f, u_train, u_test = load_data_mnist('./mnist_train.npz', device)

cin = 3
k = 20
N = 128

# q_train = torch.cat((q_train, q_train*f[0].unsqueeze(0)), dim = 1)
# q_test = torch.cat((q_test, q_test*f[0].unsqueeze(0)), dim = 1)
# u_train = u_train[:, :, 0, :, :]
# u_test = u_test[:, :, 0, :, :]

# ind0 = np.arange(0, 1000)
# ind_filter = ind0 % 100 < 50
# ind = ind0[ind_filter]

# q_train = q_train[ind]
# u_train = u_train[ind]

print(q_train.shape)
print(q_test.shape)
print(u_train.shape)
print(u_test.shape)

Batch_size = 20

train_data = Data.TensorDataset(q_train, u_train)
N_Batch = N_train/Batch_size
train_loader = Data.DataLoader(dataset = train_data, batch_size = Batch_size, shuffle = False)

test_data = Data.TensorDataset(q_test, u_test)
N_Batch = N_test/Batch_size
test_loader = Data.DataLoader(dataset = test_data, batch_size = Batch_size, shuffle = False)

modes = 12
width = 32

if operator == "nqf":
    if model_type == "UNO":
        model = Neumann_MyUNO_mnist(N, k, modes1 = modes, modes2 = modes, width = width, FNO_act = FNO_act, CNN_act = CNN_act, num_layer = num_layer, DC_in = DC_in).to(device)
    else:
        model = Neumann_FNO_mnist(N, k, modes1 = modes, modes2 = modes, width = width, FNO_act = FNO_act, CNN_act = CNN_act, DC_in = DC_in).to(device)
else:
    if model_type == "UNO":
        model = MyUNO(N, k, modes1 = modes, modes2 = modes, width = width, FNO_act = FNO_act, CNN_act = CNN_act, DC_in = DC_in, num_layer = num_layer, cin = cin).to(device)
    else:
        model = FNO2d(modes1 = modes, modes2 = modes, width = width, FNO_act = FNO_act, CNN_act = CNN_act, DC_in = DC_in, cin = cin).to(device)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

epochs = 1000
step_size = 200

gamma = 0.5
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# start_lr = 1e-3
# end_lr = 1e-5
# optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=end_lr)

train_er, test_er, PINN_train_er,  PINN_test_er = train_model_mnist(train_loader, test_loader, model, epochs, optimizer, scheduler, N, k, f,
                                                                    data_weight = data_weight, PINN_weight = PINN_weight)

ave_train_error, ave_test_error = model_test_mnist(train_loader, test_loader, model, k, f)
print(ave_train_error)
print(ave_test_error)

if save_model:
    model = model.cpu()
    torch.save(model.state_dict(), filename_model)

    np.savez(filename_data, width=width, modes=modes, times=times, operator=operator, model_type=model_type,
             FNO_act=FNO_act, CNN_act=CNN_act, cin=cin, DC_in=DC_in,
             Batch_size=Batch_size, step_size=step_size, gamma=gamma, learning_rate=lr, epochs=epochs,
             data_weight=data_weight, PINN_weight=PINN_weight, num_layer=num_layer,
             train_er=train_er, test_er=test_er, PINN_train_er=PINN_train_er, PINN_test_er=PINN_test_er,
             ave_train_error=ave_train_error, ave_test_error=ave_test_error)
    
print(filename_model)