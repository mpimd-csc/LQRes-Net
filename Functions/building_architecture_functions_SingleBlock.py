""" Network design """
import torch
import torch.nn as nn


def mykron(a):
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

def mykronCompact(a,idx=None):
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
        
    if idx == None:
        out1 = out
    else:
        out1 = out[:,idx]
    return out1


## Residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_features,activation = nn.GELU):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            self.activation(),
            nn.Linear(in_features, in_features),
        )
        
    def forward(self,x):
        # return self.block(x)
        return x + self.block(x)
    
## ResNet for nonlinear part 
class nonlinear_part(nn.Module):
    def __init__(self,n,num_residual_blocks,p=2,lb = True, activation = nn.GELU):
        super(nonlinear_part,self).__init__()
        self.activation = activation
        model = [
            nn.Linear(n, n*p),
            # self.activation(),
            # nn.Linear(n*p, n*p),
            ]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(n*p, activation = self.activation)]
            
        model += [
            # nn.Linear(n*p, n*p),
            # self.activation(),
            nn.Linear(n*p, 1,bias = lb),
            ]        
        self.model = nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)

## Build models by adding linear and quadratic shortcuts
class model_LQ_DL(nn.Module):
    def __init__(self, n,num_residual_blocks=4,p=1,nn_bias=True,activation = nn.GELU):
        super(model_LQ_DL,self).__init__()
        self.activation = activation
        
        self.linear = nn.Linear(n,1)
        # self.quad = nn.Linear(self.nq,1,bias=False)
        self.nonlinear = nonlinear_part(n,num_residual_blocks,p,activation = self.activation)
        
    def forward(self,x):
        return self.linear(x)  + self.nonlinear(x)  


## Build models using only ResNet
class model_DL(nn.Module):
    def __init__(self, n,num_residual_blocks=4,p=1,activation = nn.GELU):
        super(model_DL,self).__init__()
        self.activation = activation
        self.nonlinear = nonlinear_part(n,num_residual_blocks,p,activation = self.activation)
        
    def forward(self,x):
        return self.nonlinear(x)  
    
## Build models using only ResNet
class model_LQ_DL_New(nn.Module):
    def __init__(self, n,num_residual_blocks=4,p=1,nn_bias=True,activation = nn.GELU):
        super(model_LQ_DL_New,self).__init__()
        self.activation = activation
        
        # self.idx = [i*n+j for i in range(n) for j in range(i,n)]
        self.idx = [i for i in range(n**2)]
        self.nq = len(self.idx)
        
        self.linear = nn.Linear(n,1)
        self.quad = nn.Linear(self.nq,1,bias=False)
        self.nonlinear = nonlinear_part(n,num_residual_blocks,p,activation = self.activation)
        self.linear_out = nn.Linear(2,1)
        
    def forward(self,x):
        xl = self.linear(x)
        xq = self.quad(mykronCompact(x))
        xn = self.nonlinear(x)
        xt = torch.cat((xq,xn),dim=1)
        return self.linear_out(xt)
        # return self.nonlinear(x)
        # return self.linear(x)  + self.quad(mykronCompact(x)) #+ self.nonlinear(x)  
    
class model_DL_fully(nn.Module):
    def __init__(self, n,num_residual_blocks=4,p=1,activation = nn.GELU):
        super(model_DL_fully,self).__init__()
        self.activation = activation
        model = [
            nn.Linear(n, n*p),
            self.activation(),
            nn.Linear(n*p, n*p),
            self.activation(),
            nn.Linear(n*p, n*p),
            self.activation(),
            nn.Linear(n*p, n*p),
            self.activation(),
            nn.Linear(n*p, n*p),
            self.activation(),
            nn.Linear(n*p, n*p),
            self.activation(),
            nn.Linear(n*p, 1),
            ]
        self.model = nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)  
    
    
