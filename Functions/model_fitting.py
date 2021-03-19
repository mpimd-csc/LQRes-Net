""" Training of a network """

import torch
import sys
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable


def fit_model(model, dataloaders, criterion, optimizer, num_epochs=500, print_opt = True, plot_opt = True):

    scheduler = StepLR(optimizer, step_size=1000, gamma = 0.1)

    track_error = {'train': torch.zeros((num_epochs,)), 'test': torch.zeros((num_epochs,))}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    len_train = len(dataloaders['train'].dataset)
    len_test = len(dataloaders['test'].dataset)
    

    for i in range(num_epochs):   
        # optimizer.param_groups[0]['initial_lr'] = 0.99*optimizer.param_groups[0]['lr']
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(dataloaders['train']), T_mult=1, eta_min=1e-6, last_epoch=-1)

        loss_train = 0.0
        loss_test = 0.0      
        
        model.train() 
        for x,y in dataloaders['train']:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            
            loss = criterion(outputs,y)
                        
            
            loss_train += loss.item()*y.numel()
    
            loss.backward()
            optimizer.step()
                        
        with torch.no_grad():
            model.eval()
            for x,y in dataloaders['test']:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs,y)
                loss_test += loss.item()*y.numel()
        
        if print_opt:
            sys.stdout.write("\r[Epoch %d/%d] [Training loss: %.2e] [Testing loss: %.2e] [learning rate: %.2e]" 
                              % (i+1,num_epochs,loss_train/len_train,loss_test/len_test,optimizer.param_groups[0]['lr'],))
                       
        track_error['train'][i] = loss_train/len_train
        track_error['test'][i] = loss_test/len_test
        
        scheduler.step()


        
    
    if plot_opt:
        fig = plt.figure()
        ax = ax = plt.axes()
        plt.semilogy(track_error['train'])
        plt.semilogy(track_error['test'])
        plt.show()
            
    return model, track_error