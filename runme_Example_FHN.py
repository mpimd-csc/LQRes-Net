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


@dataclass
class parameters:
    bs: int = 512
    splitdata: float = 0.2
    num_epochs: int = 500
    lr: float = 5e-4
    save_model_path: str = './Results/FHN/'
    weightdecay: float =1e-4
    NumInitial: int = 10
    NumResBlocks: int = 2
    deri: str = 'approx' 
    training_model: bool = False

Params = parameters()

# Make a folder to save results if it does not exist
os.makedirs(os.path.dirname(Params.save_model_path), exist_ok=True)

# Set device. As if now, the code is only tested on CPU; therefore, we now enforce device to be "cpu".
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
req_plot = True

# Define the Fitz-Hugh Nagumo model
model = models.FHN_model
deri = lambda t,x: model(x,t)



# Collect simulated data for different initial conditions
n = 2
ts = np.linspace(0,200, 5000)
dt = ts[1]-ts[0]
t = np.empty((0))
x = np.empty((0,n))
dx = np.empty((0,n))

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, Params.NumInitial)]

for i in range(Params.NumInitial):
    x0 =  np.random.uniform(-1,1,(2,))
   
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


learnt_deri = []

fig, ax = plt.subplots(1,1)

start_time = time.time()


for i in range(n):
    print('='*75)
    dx0 = dx[:,i]

    if e2[i] < 1e-5:
        print('For {} component: Linear and quadratic terms are sufficient. \nHence, we do not need build a neural network \nbut rather directly determine linear and quadratic terms.\n'.format(i))
        U = utlis.quad_function_deri(dx0,x)
        f1 =  lambda z: np.concatenate((z, utlis.def_kron(torch.tensor(z)).numpy(),np.ones((1,))))@U
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
            torch.save(net_LQ_DL,'./Results/FHN/FHN_Modelcomp_{}.pkl'.format(i))
            torch.save(err_LQ_DL,'./Results/FHN/FHN_Errcomp_{}.pkl'.format(i))
        else:
            print('Loading pre-trained model')
            net_LQ_DL = torch.load('./Results/FHN/FHN_Modelcomp_{}.pkl'.format(i))
            err_LQ_DL = torch.load('./Results/FHN/FHN_Errcomp_{}.pkl'.format(i))
            
        
        print('\n')
        f1 = lambda z: net_LQ_DL(torch.tensor(z).unsqueeze(dim=0).float()).squeeze(dim=0).detach().numpy().item()
        ax.loglog(err_LQ_DL['train'],'b', label = 'traiing using NN for {} component'.format(i))
        ax.loglog(err_LQ_DL['test'],'r--', label = 'traiing using NN for {} component'.format(i))
        ax.legend()

    
    learnt_deri.append(f1)
    
print('='*75)
print('Training time')
print("%.2s seconds" % (time.time() - start_time))
Compute_deri = lambda z,t: np.array([d(z) for d in learnt_deri])

x0 =  np.random.uniform(-1,1,(2,))
ts = np.linspace(0,1000, 10000)

print('='*75)
print('Simulating models....')

# Solve the differential equations both grouth truth and learn equations
sol = odeint(model, x0, ts)
sol_NN = odeint(Compute_deri, x0, ts)

print('Plotting the results....')

fig, ax = plt.subplots(1,1)
ax.plot(ts,sol)
fig, ax = plt.subplots(1,1)
ax.plot(ts,sol_NN)

# Plot
fig = plt.figure(figsize=(22,4))

ax = plt.subplot(121)
ax.plot(sol[:,0], sol[:,1], lw=2)
ax.set_xlabel("$v$")
ax.set_ylabel("$w$")
ax.set_title("Known Model")

ax = plt.subplot(122)
ax.plot(sol_NN[:,0], sol_NN[:,1], lw=2)
ax.set_xlabel("$v$")
ax.set_ylabel("$w$")
ax.set_title("LQRes-Net")

tikzplotlib.save(Params.save_model_path + "FHN_2D_limit.tex")
plt.show()


# Plot the limit-cycles
fig = plt.figure(figsize=(15,8))

ax = plt.subplot(111,projection='3d')
plt.gca().patch.set_facecolor('white')

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_ylim((-1.75,2))

ax.plot(ts[0::1], sol[0::1,0], sol[0::1,1], lw=3, label = 'True Model')
ax.plot(ts[0::5], sol_NN[0::5,0], sol_NN[0::5,1], lw=2,marker = "8", linestyle = 'None',label = 'LQResNet Model')
ax.legend(bbox_to_anchor=(0.75, 1.05))

ax.set_xlabel("Time")
ax.set_ylabel("$v$")
ax.set_zlabel("$w$")

tikzplotlib.save(Params.save_model_path + "FHN_3D_Comparison.tex")
plt.show()
fig.savefig(Params.save_model_path + "FHN_3D_Comparison.pdf", bbox_inches = 'tight',
    pad_inches = 0)






