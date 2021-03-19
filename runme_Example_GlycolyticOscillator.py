#!/usr/bin/env python

# Import necessary packages
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch_optimizer as optim_all
import tikzplotlib
import matplotlib.pyplot as plt
import Functions.models as models
import Functions.model_fitting as model_fitting
import Functions.architecture_design as arch_design
import Functions.utlis as utlis
from Functions.Approx_derivative import xdot
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(seed=42)


# Define parameters
@dataclass
class parameters:
    bs: int = 512
    splitdata: float = 0.2
    num_epochs: int = 2000
    lr: float = 1e-3
    save_model_path: str = './Results/Glycolytic_Oscillator/'
    weightdecay: float =1e-4
    NumInitial: int = 30
    NumResBlocks: int = 5
    deri: str = 'approx' 
    training_model: bool = False

Params = parameters()

# Make a folder to save results if it does not exist
os.makedirs(os.path.dirname(Params.save_model_path), exist_ok=True)

# Set device. As if now, the code is only tested on CPU; therefore, we now enforce device to be "cpu".
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
req_plot = True

# Define the Glycolytic Oscillator model
model = models.GlycolyticOsci_model
deri = lambda t,x: model(x,t)


# Collect simulated data for different initial conditions
ts = np.linspace(0,10, 4000)
dt = ts[1]-ts[0]

n = 7
t = np.empty((0))
x = np.empty((0,n))
dx = np.empty((0,n))


for i in range(Params.NumInitial):
    xmin = np.array([0.15,0.19,0.04,0.10,0.08,0.14,0.05])
    xmax = np.array([1.60,2.16,0.20,0.35,0.30,2.67,0.10])
    x0 =  xmin + np.random.uniform(0,1,(n,))*(xmax-xmin)
    
    # Solve the equation
    sol = odeint(model, x0, ts)
    t = np.concatenate((t, ts), axis=0)
    x = np.concatenate((x, sol), axis=0)


    if Params.deri == 'exact':
        dx = np.concatenate((dx,[deri(i,j) for i,j in zip(ts,sol)]), axis=0)
    else:
        dx = np.concatenate((dx,xdot(sol.T, dt, order=4).T), axis=0)


# Identify if the quadratic model is sufficent to learn the dynamics. 
# If yes, then we do not require a neural network to learn any dynamics.
e1,e2 = utlis.error_linear_quad(dx,x)
fig, ax = plt.subplots(2,1)
ax[0].semilogy(e1,'b+',label = 'linear')
ax[0].semilogy(e2,'r+',label = 'quadratic')
ax[0].legend()
ax[1].semilogy((e2/e1)*(e1>1e-8),'b+',label = 'improvement due to quadratic term')
ax[1].legend()



# Learn dynamics using LQResNet when the quadratic form is not sufficient enough.
learnt_deri = []
start_time = time.time()
fig, ax = plt.subplots(1,1)
for i in range(n):
    print('='*75)
    dx0 = dx[:,i]

    if e2[i] < 5e-4:
        print('For {} component: Linear and quadratic terms are sufficient. \nHence, we do not need build a neural network \nbut rather directly determine linear and quadratic terms.\n'.format(i))
        U = utlis.quad_function_deri(dx0,x)
        f1 =  lambda z,U=U: np.concatenate((z, utlis.def_kron(torch.tensor(z)).numpy(),np.ones((1,))))@U
    else:
        print('For {} component: Linear and quadratic terms are NOT sufficient. \nHence, we train a neural network.\n'.format(i))
        # Define the training and testing data, and also convert the data-type to torch.tensor()
        x_train, x_test, dx_train, dx_test = train_test_split(x, dx0, 
                                                    test_size= Params.splitdata, random_state=42)
        x_train = torch.tensor(x_train).float()
        dx_train = torch.tensor(dx_train).unsqueeze(dim=1).float()
        x_test = torch.tensor(x_test).float()
        dx_test = torch.tensor(dx_test).unsqueeze(dim=1).float()
        
        # Zipping the x and dx together for training and testing data
        train_dset = list(zip(x_train,dx_train))
        test_dset = list(zip(x_test,dx_test))
        
        # Define dataloaders
        train_dl = torch.utils.data.DataLoader(train_dset, batch_size =Params.bs)
        test_dl = torch.utils.data.DataLoader(test_dset, batch_size = Params.bs)
        dataloaders = {'train': train_dl, 'test': test_dl}
        
        net_LQ_DL = arch_design.LQResNet(n,num_residual_blocks=Params.NumResBlocks, p = 5, activation = nn.ELU)

        # Load the model to device
        net_LQ_DL = net_LQ_DL.to(device)
        
        # Define criterion to define the loss function
        criterion = nn.MSELoss()
        
        # Define the Optimizer
        opt_func_LQ_DL = optim_all.RAdam(net_LQ_DL.parameters(), lr = Params.lr,weight_decay=Params.weightdecay) 
        
        if Params.training_model:
            print('Training Res-network...')
            net_LQ_DL, err_LQ_DL = model_fitting.fit_model(net_LQ_DL, dataloaders, criterion, opt_func_LQ_DL, num_epochs=Params.num_epochs ,plot_opt=False)
            # Saving the network
            torch.save(net_LQ_DL,'./Results/Glycolytic_Oscillator/GO_Modelcomp_{}.pkl'.format(i))
            torch.save(err_LQ_DL,'./Results/Glycolytic_Oscillator/GO_Errcomp_{}.pkl'.format(i))
        else:
            print('Loading pre-trained model')
            net_LQ_DL = torch.load('./Results/Glycolytic_Oscillator/GO_Modelcomp_{}.pkl'.format(i))
            err_LQ_DL = torch.load('./Results/Glycolytic_Oscillator/GO_Errcomp_{}.pkl'.format(i))
            
            
        print('\n')
        f1 = lambda z,net_LQ_DL=net_LQ_DL: net_LQ_DL(torch.tensor(z).unsqueeze(dim=0).float()).squeeze(dim=0).detach().numpy().item()
        ax.loglog(err_LQ_DL['train'],'b', label = 'traiing using NN for {} component'.format(i))
        ax.loglog(err_LQ_DL['test'],'r--', label = 'traiing using NN for {} component'.format(i))
        ax.legend()
        
    learnt_deri.append(f1)
    
print('='*75)
print('Training time')
print("%.5f seconds" % (time.time() - start_time))
                 
Compute_deri = lambda z,t: np.array([d(z) for d in learnt_deri])

print('='*75)
print('Simulating models....')
xmin = np.array([0.15,0.19,0.04,0.10,0.08,0.14,0.05])
xmax = np.array([1.60,2.16,0.20,0.35,0.30,2.67,0.10])
x0 =  xmin + np.random.uniform(0,1,(n,))*(xmax-xmin)   

ts = np.linspace(0,10, 5000)

# Solve the differential equations both grouth truth and learn equations
sol = odeint(model, x0, ts)
sol_NN = odeint(Compute_deri, x0, ts)

# Plot the results
fig, ax = plt.subplots(1,1)
ax.plot(ts,sol, color = 'r', lw = 3,label = 'True Model')
ax.plot(ts[0::15],sol_NN[0::15,:], linestyle = 'None', marker = '8',color = 'c', lw = 2,label = 'LQRes-Net Model')
ax.set_xlabel('Time')
ax.set_ylabel('All $S$ components')
ax.legend()

tikzplotlib.save(Params.save_model_path + "GO_all_components.tex")
plt.show()

fig.savefig(Params.save_model_path + "GO_all_components.pdf")


# Plot the limit-cycles
fig = plt.figure(figsize=(4,4))

ax = plt.subplot(111)
ax.plot(sol[:,4], sol[:,5], lw=2,label = 'True Model')
ax.plot(sol_NN[:,4], sol_NN[:,5],lw=2,linestyle = '--',label = 'LQR-Net Model')
ax.set_xlabel("$S_4$")
ax.set_ylabel("$S_5$")
ax.legend()

tikzplotlib.save(Params.save_model_path + "GO_2D_limit_S4S5.tex")
plt.show()

fig.savefig(Params.save_model_path + "GO_2D_limit_S4S5.pdf")



# Plot the limit-cycles, reported in the paper

fig = plt.figure(figsize=(45,30))

ax = fig.add_subplot(2,2,1,projection='3d')
ax.plot(ts, sol[:,0], sol[:,1], lw=2,label = 'True Model')
ax.plot(ts, sol_NN[:,0], sol_NN[:,1], lw=2, linestyle = '--',label = 'LQResNet Model', color = 'm')
ax.set_xlabel("Time")
ax.set_ylabel("$S_0$")
ax.set_zlabel("$S_1$")
ax.legend()

ax = fig.add_subplot(2,2,2,projection='3d')
ax.plot(ts, sol[:,0], sol[:,2], lw=2,label = 'True Model')
ax.plot(ts, sol_NN[:,0], sol_NN[:,2], lw=2, linestyle = '--',label = 'LQResNet Model', color = 'm')
ax.set_xlabel("Time")
ax.set_ylabel("$S_0$")
ax.set_zlabel("$S_2$")
ax.legend()

ax = fig.add_subplot(2,2,3,projection='3d')
ax.plot(ts, sol[:,0], sol[:,3], lw=2,label = 'True Model')
ax.plot(ts, sol_NN[:,0], sol_NN[:,3], lw=2, linestyle = '--',label = 'LQResNet Model', color = 'm')
ax.set_xlabel("Time")
ax.set_ylabel("$S_0$")
ax.set_zlabel("$S_3$")
ax.legend()

ax = fig.add_subplot(2,2,4,projection='3d')
ax.plot(ts, sol[:,0], sol[:,4], lw=2,label = 'True Model')
ax.plot(ts, sol_NN[:,0], sol_NN[:,4], lw=2, linestyle = '--',label = 'LQResNet Model', color = 'm')
ax.set_xlabel("Time")
ax.set_ylabel("$S_0$")
ax.set_zlabel("$S_4$")
ax.legend()

fig.tight_layout(pad=-4.0)


tikzplotlib.save(Params.save_model_path + "GO_3D_plots.tex")
plt.show()

fig.savefig(Params.save_model_path + "GO_3D_plots.pdf")





