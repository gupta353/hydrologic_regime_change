
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import choices
import copy
import random
import HydroMetrics

import torch
from torch import classes, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def LSTMImplement(train_rel, val_rel, test_rel, batch_size, input_dim, hidden_dim, n_layers, output_dim, epochs, seeds, device, save_direc, numTrainSamps):
    # train_rel, val_rel, and test_rel = lists containing training, validation, and test data; each element of the list contains data for one basin 

    ####################################################################################################
    # LSTM classes
    ########################################################################################################
    # build a dataset class
    class CustomData(Dataset):
        def __init__(self, data, seq_len, mean_X, std_X):
            super(Dataset, self).__init__()
            self.data = data
            self.L = seq_len    # sequence length
            self.mx = mean_X    # mean of predictor variables
            self.sdx = std_X    # standard deviation of predictor variables
            self.index_map = {}

            # create a mapping between the 2-D indices of (basin, day) to 1-D indices
            ind = 0
            for basin_ind in range(len(data)):
                for tind in range(len(data[basin_ind]) - self.L):
                    self.index_map[ind] = (basin_ind, tind)
                    ind += 1 
            
        def __len__(self):
            return len(self.index_map)
        
        def __getitem__(self, idx):
            basin_ind, tind = self.index_map[idx]
            x1 = self.data[basin_ind][tind : tind+self.L, 2:-1]     # '-1' is used to exclude the SD of streamflow as predictor
            x1 = torch.div(x1 - self.mx, self.sdx)                  # normalization of the variable
            y1 = torch.div(self.data[basin_ind][tind+self.L-1 ,1], self.data[basin_ind][tind+self.L-1 , -1])
            return x1, y1

    # define LSTM model class
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
            super(LSTMModel, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first = True)
            self.fc = nn.Linear(hidden_dim, 1)                      # fully connected layer
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p = 0.40)

        def forward(self, x):                                       # x can contain multiple samples
            h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
            c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0, c0))
            #out = self.fc(out[:,-1,:]) #use hn instead, use relu before fc
            out = self.fc(self.dropout(hn[0,:,:]))
            #out = self.fc(hn[0,:,:])
            #out=self.fc1(out)
            return out

    # define the module to train the model
    def train_mod(dataloader, model, loss_fn, optimizer):
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        model.train()
        
        tr_loss  = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y.view(len(y),1))          # .view is used to reshape the y valus

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss

        tr_loss /= size
        return tr_loss, model.state_dict()

    # define the module to validate the model
    def validate_mod(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y.view(len(y),1))
        val_loss /= size
        print(f"Avg loss: {val_loss:>8f} \n")
        return val_loss.item()

    # define the module to test the model
    def test_mod(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        sse = 0
        ynse = []
        pred_list = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y = y.view(len(y),1)
                pred = model(X)
                test_loss += loss_fn(pred, y)
                sse += torch.sum((pred - y)**2)
                ynse.append(y)
                pred_list.append(pred)
        ynse = torch.cat(ynse)
        pred_list = torch.cat(pred_list)
        sst = torch.sum((ynse - torch.mean(ynse))**2)
        nse = 1 - sse/sst
        print(f"NSE: {nse:>8f} \n")
        return nse, pred_list, ynse
##############################################################################################################################

    # finding mean and SD of predictor variables for normalization 
    train_data_comb = np.concatenate(train_rel)
    mean_X = np.mean(train_data_comb[:,2:-1], axis = 0)
    std_X = np.std(train_data_comb[:,2:-1], axis = 0)

    # convert data to torch tensors
    train_ten = []
    for train_tmp in train_rel:
        train_ten.append(torch.from_numpy(train_tmp).float())

    val_ten = []
    for val_tmp in val_rel:
        val_ten.append(torch.from_numpy(val_tmp).float())

    test_ten = []
    for test_tmp in test_rel:            #################
        test_ten.append(torch.from_numpy(test_tmp).float())

    mean_X = torch.from_numpy(mean_X).float()
    std_X = torch.from_numpy(std_X).float()
########################################################################################################################################################
    
    # define dataloader
    train_dataset = CustomData(train_ten, 365, mean_X, std_X)
    train_dataloader  = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    val_dataset = CustomData(val_ten, 365, mean_X, std_X)
    val_dataloader  = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, drop_last = True)
    
    test_obs_pred_seed = []
    for seed in seeds:

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # define LSTM model
        lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
        lstm.cuda()

        # define loss and optimizer
        lossfn = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr = 10**(-3))

        # fix the number of epochs and start model training
        loss_tr_list = []
        loss_vl_list = []
        model_state = []
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            loss_tr, state = train_mod(train_dataloader, lstm, lossfn, optimizer)
            loss_vl = validate_mod(val_dataloader, lstm, lossfn)
        
            model_state.append(copy.deepcopy(state))
            loss_tr_list.append(loss_tr)
            loss_vl_list.append(loss_vl)
        minind = np.nonzero(loss_vl_list == np.min(loss_vl_list))
        minind = minind[0][0]
        lstm.load_state_dict(model_state[minind], strict = True)
        
        # save model
        fname = 'model_save_seed_' + str(seed) + '_num_' + str(numTrainSamps)
        filename = save_direc + '/' + fname
        torch.save(lstm.state_dict(), filename)

        # Make predictions on test set
        NSE = []
        test_obs_pred = []
        ind = 0
        for test_tmp in test_ten:
            # create test dataloader
            test_dataset = CustomData([test_tmp], 365, mean_X, std_X)
            g = torch.Generator()
            g.manual_seed(0)
            test_dataloader  = DataLoader(test_dataset, batch_size = 256, shuffle=False, generator = g)
            # make predictions
            nse, ypred, yobs = test_mod(test_dataloader, lstm, lossfn)
            yobs = yobs.cpu().detach().numpy()
            ypred = ypred.cpu().detach().numpy()
            NSE.append(nse)
            test_obs_pred.append([yobs, ypred])
            ind += 1
        test_obs_pred_seed.append(test_obs_pred)

    # compute average prediction obtained from different seeds
    obs_pred_avg  = []
    for basin_ind in range(len(test_rel)):
        ypred = 0
        for seed_ind in range(len(seeds)):
            ypred += test_obs_pred_seed[seed_ind][basin_ind][1]
        ypred /= len(seeds)
        yobs = test_obs_pred_seed[seed_ind][basin_ind][0]
        obs_pred_avg.append([yobs, ypred])
    
    # compute NSE of average prediction
    NSE_avg = []
    for obs_pred in obs_pred_avg:
        nse = HydroMetrics.computeNSE(obs_pred[0], obs_pred[1])
        NSE_avg.append(nse)

    # compute uncertainty in NSE
    NSEBS = []
    for obs_pred in obs_pred_avg:
        nse = HydroMetrics.computeNSEBS(obs_pred[0], obs_pred[1])
        NSEBS.append(nse)

    return NSE_avg, NSEBS
