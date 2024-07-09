from mumps import DMumpsContext
# from utils import *
import numpy as np
from scipy.sparse import dia_matrix

ctx = DMumpsContext()
ctx.set_silent()

def Matrix_Gen(N, Q, k, thickness=0.05, gamma = 2.8, Transpose=False):   
    h = 1/N
    constant = k
    N_PML = int(N*thickness)
    N_total = N + 2*N_PML
    M = N_total-1
    q_total = np.pad(Q.reshape(N+1,N+1),N_PML)
    A = np.ones((N_total,N_total+1),dtype = np.complex128)
    for j in range(N_PML):
        A[:,j] -= 1j*constant*(N_PML-j)**2/(N_PML**2)
        A[:,-j-1] -= 1j*constant*(N_PML-j)**2/(N_PML**2)
    for i in range(N_PML):
        A[i] /= (1-1j*constant*(N_PML-i-0.5)**2/(N_PML**2))
        A[-i-1] /= (1-1j*constant*(N_PML-i-0.5)**2/(N_PML**2))
    B = A.T
    C = np.ones((N_total+1,N_total+1),dtype = np.complex128)
    for i in range(N_PML):
        C[i] -= 1j*constant*(N_PML-i)**2/(N_PML**2)
        C[-i-1] -= 1j*constant*(N_PML-i)**2/(N_PML**2)
    for j in range(N_PML):
        C[:,j] *= (1-1j*constant*(N_PML-j)**2/(N_PML**2))
        C[:,-j-1] *= (1-1j*constant*(N_PML-j)**2/(N_PML**2))
    C *= k*k*(1+q_total)
    M0 = (2/3+gamma/36)*h*h*C[1:-1,1:-1] - 5/6*(A[:-1,1:-1]+A[1:,1:-1]+B[1:-1,1:]+B[1:-1,:-1])
    m0 = np.tile(M0.reshape(-1,),2)[0:0+M*M]
    M1 = 1/12*(1-gamma/6)*h*h*C[:-2,1:-1] + 5/6*A[:-1,1:-1] - 1/12*(B[:-2,:-1]+B[:-2,1:])
    M1[0] *= 0
    m1 = np.tile(M1.reshape(-1,),2)[M:M+M*M]
    M2 = 1/12*(1-gamma/6)*h*h*C[2:,1:-1] + 5/6*A[1:,1:-1] - 1/12*(B[2:,:-1]+B[2:,1:])
    M2[-1] *= 0
    m2 = np.tile(M2.reshape(-1,),2)[M*M-M:M*M-M+M*M]
    M3 = 1/12*(1-gamma/6)*h*h*C[1:-1,:-2] + 5/6*B[1:-1,:-1] - 1/12*(A[:-1,:-2]+A[1:,:-2])
    M3[:,0] *= 0
    m3 = np.tile(M3.reshape(-1,),2)[1:1+M*M]
    M4 = 1/12*(1-gamma/6)*h*h*C[1:-1,2:] + 5/6*B[1:-1,1:] - 1/12*(A[:-1,2:]+A[1:,2:])
    M4[:,-1] *= 0
    m4 = np.tile(M4.reshape(-1,),2)[M*M-1:M*M-1+M*M]
    M5 = 1/12*(A[:-1,:-2]+B[:-2,:-1]) + gamma*h*h/144*C[:-2,:-2]
    M5[0] *= 0
    M5[:,0] *= 0
    m5 = np.tile(M5.reshape(-1,),2)[M+1:M+1+M*M]
    M6 = 1/12*(A[1:,:-2]+B[2:,:-1])+ gamma*h*h/144*C[2:,:-2]
    M6[-1] *= 0
    M6[:,0] *= 0
    m6 = np.tile(M6.reshape(-1,),2)[M*M-M+1:M*M-M+1+M*M]
    M7 = 1/12*(A[:-1,2:]+B[:-2,1:])+ gamma*h*h/144*C[:-2,2:]
    M7[0] *= 0
    M7[:,-1] *= 0
    m7 = np.tile(M7.reshape(-1,),2)[M-1:M-1+M*M]
    M8 = 1/12*(A[1:,2:]+B[2:,1:])+ gamma*h*h/144*C[2:,2:]
    M8[-1] *= 0
    M8[:,-1] *= 0
    m8 = np.tile(M8.reshape(-1,),2)[M*M-M-1:M*M-M-1+M*M]
    if Transpose:
        data = (np.c_[np.tile(m0,2).real,np.tile(m1,2).real,np.tile(m2,2).real,
                      np.tile(m3,2).real,np.tile(m4,2).real,np.tile(m5,2).real,
                      np.tile(m6,2).real,np.tile(m7,2).real,np.tile(m8,2).real,
                      -np.tile(m0,2).imag,-np.tile(m1,2).imag,-np.tile(m2,2).imag,
                      -np.tile(m3,2).imag,-np.tile(m4,2).imag,-np.tile(m5,2).imag,
                      -np.tile(m6,2).imag,-np.tile(m7,2).imag,-np.tile(m8,2).imag,
                      np.tile(m0,2).imag,np.tile(m1,2).imag,np.tile(m2,2).imag,
                      np.tile(m3,2).imag,np.tile(m4,2).imag,np.tile(m5,2).imag,
                      np.tile(m6,2).imag,np.tile(m7,2).imag,np.tile(m8,2).imag]).transpose()
    else:
        data = (np.c_[np.tile(m0,2).real,np.tile(m1,2).real,np.tile(m2,2).real,
              np.tile(m3,2).real,np.tile(m4,2).real,np.tile(m5,2).real,
              np.tile(m6,2).real,np.tile(m7,2).real,np.tile(m8,2).real,
              np.tile(m0,2).imag,np.tile(m1,2).imag,np.tile(m2,2).imag,
              np.tile(m3,2).imag,np.tile(m4,2).imag,np.tile(m5,2).imag,
              np.tile(m6,2).imag,np.tile(m7,2).imag,np.tile(m8,2).imag,
              -np.tile(m0,2).imag,-np.tile(m1,2).imag,-np.tile(m2,2).imag,
              -np.tile(m3,2).imag,-np.tile(m4,2).imag,-np.tile(m5,2).imag,
              -np.tile(m6,2).imag,-np.tile(m7,2).imag,-np.tile(m8,2).imag]).transpose()
    offsets = np.array([0,
                        -M, M,
                        -1, 1,
                        -M - 1, M - 1,
                        -M + 1, M + 1,
                        M*M,
                        M*M-M, M*M+M,
                        M*M-1, M*M+1,
                        M*M-M - 1, M*M+M - 1,
                        M*M-M + 1, M*M+M + 1,
                        -M*M,
                        -M*M-M, -M*M+M,
                        -M*M-1, -M*M+1,
                        -M*M-M - 1, -M*M+M - 1,
                        -M*M-M + 1, -M*M+M + 1])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    mat = dia.tocoo()
    if Transpose:
        mat = mat.T
    return mat
    
def Matrix_analysis(N, Q, k = 20, thickness = 0.05, Transpose=False):
    global ctx  
    _Matrix_ = Matrix_Gen(N, Q, k, thickness, Transpose)
    ctx.set_shape(_Matrix_.shape[0])
    if ctx.myid == 0:
        ctx.set_centralized_assembled_rows_cols(
            _Matrix_.row + 1, _Matrix_.col + 1)
    ctx.run(job = 1)
    
def F_laplacian(F):
    N = int(np.sqrt(F.shape[0])) - 1
    f = F.reshape((N+1,N+1))
    f1 = np.zeros_like(f)
    f1[1:-1,1:-1] = f[2:,1:-1] + f[:-2,1:-1] + f[1:-1,:-2] + f[1:-1,2:] - 4*f[1:-1,1:-1]
    f1[0,0] = 4*f[0,0]-5*f[1,0]+4*f[2,0]-f[3,0]-5*f[0,1]+4*f[0,2]-f[0,3]
    f1[0,-1] = 4*f[0,-1]-5*f[1,-1]+4*f[2,-1]-f[3,-1]-5*f[0,-2]+4*f[0,-3]-f[0,-4]
    f1[-1,0] = 4*f[-1,0]-5*f[-2,0]+4*f[-3,0]-f[-4,0]-5*f[-1,1]+4*f[-1,2]-f[-1,3]
    f1[-1,-1] = 4*f[-1,-1]-5*f[-2,-1]+4*f[-3,-1]-f[-4,-1]-5*f[-1,-2]+4*f[-1,-3]-f[-1,-4]
    f1[0,1:-1] = f[0,:-2]+f[0,2:]-5*f[1,1:-1]+4*f[2,1:-1]-f[3,1:-1]
    f1[-1,1:-1] = f[-1,:-2]+f[-1,2:]-5*f[-2,1:-1]+4*f[-3,1:-1]-f[-4,1:-1]
    f1[1:-1,0] = f[:-2,0]+f[2:,0]-5*f[1:-1,1]+4*f[1:-1,2]-f[1:-1,3]
    f1[1:-1,-1] = f[:-2,-1]+f[2:,-1]-5*f[1:-1,-2]+4*f[1:-1,-3]-f[1:-1,-4]
    return f1.reshape(-1)


def Matrix_factorize(N, Q, k, thickness=0.05, Transpose=False):
    # Q:((N+1)**2,)
    global ctx
    _Matrix_ = Matrix_Gen(N, Q, k, thickness=thickness, Transpose=Transpose)
    ctx.set_centralized_assembled_values(_Matrix_.data)
    ctx.run(job = 2)
    return


def Matrix_solve(F: np.ndarray, one_dim = True, thickness=0.05, Transpose=False):
    # F:((N+1)**2,) --> ((N+1)**2,)
    global ctx
    M = int(np.sqrt(F.shape[0]))
    N = M-1
    h = 1/N
    N_PML = int(N*thickness)
    N_total = N + 2*N_PML
    F = np.pad(F.reshape(M,M),N_PML)
    if not Transpose:
        F = h*h/3*(F[2:,1:-1]+F[:-2,1:-1]+F[1:-1,2:]+F[1:-1,:-2]-F[1:-1,1:-1]).reshape(-1,)
    else:
        F = F[1:-1,1:-1]
    F = np.append(F.real, F.imag)
    _Right_ = F
    x = _Right_.copy()
    ctx.set_rhs(x)
    ctx.run(job = 3)
    tmp = x.reshape(-1, )
    output = (tmp[:(N_total-1)**2] + 1j * tmp[(N_total-1)**2:]).reshape((N_total-1),(N_total-1))
    output = np.pad(output,1)
    if Transpose:
        output[1:-1,1:-1] = (h*h)/3*(output[2:,1:-1]+output[:-2,1:-1]+output[1:-1,2:]+output[1:-1,:-2]-output[1:-1,1:-1])
    output = output[N_PML:-N_PML,N_PML:-N_PML]
    if one_dim:
        output = output.reshape(-1,)
    return output
    