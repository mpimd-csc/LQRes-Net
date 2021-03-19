import scipy.linalg as spla
import numpy as np
import torch
# from building_architecture_functions_SingleBlock import *

def def_kron(a):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    if len(a.shape) > 3:
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(a.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * a.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
    else:
        a = a.unsqueeze(dim=1)
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(a.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * a.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1).squeeze(dim=1)
    return out

def error_linear_quad(dx,x):
    x_l = np.concatenate((x, np.ones((x.shape[0],1))),axis=1)
    Ab_dir = spla.lstsq(x_l,dx)
    Rlx =  dx - x_l@Ab_dir[0]
    
    x_q = np.concatenate((x, def_kron(torch.tensor(x)).numpy(),np.ones((x.shape[0],1))),axis=1)
    AHb_dir = spla.lstsq(x_q,dx)
    Rqx =  dx - x_q@AHb_dir[0]
    
    e1 = np.mean(np.abs(Rlx),axis=0)    
    e2 = np.mean(np.abs(Rqx),axis=0)    
    
    return e1,e2


def quad_function_deri(dx,x):
    x_q = np.concatenate((x, def_kron(torch.tensor(x)).numpy(),np.ones((x.shape[0],1))),axis=1)
    AHb_dir = spla.lstsq(x_q,dx)
    return AHb_dir[0]