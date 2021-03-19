import numpy as np
import scipy.linalg as spla
import scipy.integrate as spint
import scipy.interpolate as spip
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt

def lorenz_model(x,t):
    dx1 = 10*(x[1] - x[0])
    dx2 = x[0]*(28-x[2]) - x[1]
    dx3 = x[0]*x[1] - 2.667*x[2]
    return np.array([dx1,dx2,dx3])


def pendulum_model(x,t,alpha = .1):
    dx1 = x[1]
    dx2 = -alpha * x[1] -np.sin(x[0])
    return np.array([dx1,dx2])



def linear_model(x,t):
    dx1 = -x[1] - 1*x[0] 
    dx2 = x[0] - 2*x[1] + 0.5*x[2]
    dx3 = -2*x[2] - 1*x[0]
    return np.array([dx1,dx2,dx3])

def artificial_model(x,t):
    dx1 = np.exp(-0.1*x[0]-1) - 1*x[0] 
    return np.array([dx1])

def quad_model(x,t):
    dx1 = -x[1] - 1*x[0]  + 0.25*x[0]*x[1]
    dx2 = -x[1] - 0.5*x[1]*x[0]
    dx3 = -2*x[2] - 1*x[0] - 1*x[2]*x[0]
    return np.array([dx1,dx2,dx3])

def FHN_model(x,t):
    a,b,g, I = (0.8, 0.7, 1/25, 0.5)
    dx1 = x[0] - (x[0]**3)/3 - x[1] + I
    dx2 = g*(x[0] + a - b*x[1])
    return np.array([dx1,dx2])

def GlycolyticOsci_model(x,t):
    k1,k2,k3,k4,k5,k6 = (100.0, 6.0, 16.0, 100.0, 1.28, 12.0)
    j0,k,kappa,q,K1, phi, N, A = (2.5,1.8, 13.0, 4.0, 0.52, 0.1, 1.0, 4.0)
    ##
    dx0 = j0 - (k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q)
    
    dx1 = (2*k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q) - k2*x[1]*(N-x[4]) - k6*x[1]*x[4]
    
    dx2 = k2*x[1]*(N-x[4]) - k3*x[2]*(A-x[5])
    
    dx3 = k3*x[2]*(A-x[5]) -k4*x[3]*x[4] - kappa*(x[3]- x[6])
    
    dx4 = k2*x[1]*(N-x[4]) -k4*x[3]*x[4] - k6*x[1]*x[4]
    
    dx5 = -(2*k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q) + 2*k3*x[2]*(A-x[5]) - k5*x[5]
    
    dx6 = phi*kappa*(x[3]-x[6]) - k*x[6]
    
    return np.array([dx0,dx1,dx2,dx3,dx4,dx5,dx6])




def solve_ode(fun_deri, x0,t):
    solver = spint.ode(fun_deri)
    solver.set_integrator('vode', method='bdf', order=5, atol=1e-8, rtol=1e-8)
    t0 = 0
    solver.set_initial_value(x0, t0)
    sol_t = [t0]
    sol_x = [x0]
    while solver.t < t:
        solver.integrate(t, step=True)
        sol_t.append(solver.t)
        sol_x.append(solver.y)
    sol_t = np.array(sol_t)
    sol_x = np.vstack(sol_x)
    
    return (sol_t,sol_x)


def normalize_data(data, dataprop):
    # the function normalize the data for given mean and std
    return (data - dataprop.mean)/dataprop.std

def dnormalize_data(data, dataprop):
    # the function dnormalize the data for given mean and std
    return (data*dataprop.std) + dataprop.mean

class Compute_Mean_Std():
    def __init__(self,data,p = False):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        if p:
            print('_'*30)
            print('mean: {}'.format(self.mean))
            print('std: {}'.format(self.std))
        
