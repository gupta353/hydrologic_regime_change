"""

Benchmark LSTM model for prediction in time using data across several watersheds 

Author: Abhinav Gupta (Created: 2 May 2022)

"""
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import choices
import copy
import random

import torch
from torch import classes, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
save_subdir = 'model_exp2'      #########

# function to read dynamic data
def readDynamic(filename):
    fid = open(filename)
    data = fid.readlines()
    fid.close()
    dyn = []
    for rind in range(1,len(data)):
        data_tmp = data[rind].split()
        yy = int(data_tmp[0])
        mm = int(data_tmp[1])
        dd = int(data_tmp[2])
        datenum = datetime.date(yy, mm, dd).toordinal()
        dyn.append([datenum] + [float(x) for x in data_tmp[3:]])
    return dyn

# function to compute NSE
def computeNSE(obs, pred):
    sse = np.sum((obs - pred)**2)
    sst = np.sum((obs - np.mean(obs))**2)
    nse = 1 - sse/sst
    return nse

# function to compute NSE uisng bootstap sampling
def computeNSEBS(obs, pred):
    counter = list(np.arange(0,len(obs)))
    nse = []
    for bs in range(1000):
        inds = choices(counter, k = len(obs))
        obs_tmp = obs[np.array(inds)]
        pred_tmp = pred[np.array(inds)]
        sse = np.sum((obs_tmp - pred_tmp)**2)
        sst = np.sum((obs_tmp - np.mean(obs_tmp))**2)
        nse.append(1 - sse/sst)
    return nse

# function to remove rows containing NaNs
def removeNaN(Y):
    X = Y.copy()
    del_ind = []
    for ind in range(0,len(X)):
        X_tmp = X[ind,:]
        if np.any(np.isnan(X_tmp)):
            del_ind.append(ind)
            
    if len(del_ind) != 0:
        Xp = np.delete(X, np.array(del_ind), axis = 0)
    else:
        Xp = X
    return Xp

########################################################################################################
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
        x1 = torch.div(x1 - self.mx, self.sdx)
        y1 = torch.div(self.data[basin_ind][tind+self.L-1 ,1], self.data[basin_ind][tind+self.L-1 , -1])
        #y1 = self.data[basin_ind][tind ,1]
        return x1, y1

# define LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.40)

    def forward(self, x):
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
    model.train()
    
    tr_loss  = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.view(len(y),1))

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
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.view(len(y),1)).item()
    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

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
            test_loss += loss_fn(pred, y).item()
            sse += torch.sum((pred - y)**2)
            ynse.append(y)
            pred_list.append(pred)
    ynse = torch.cat(ynse)
    pred_list = torch.cat(pred_list)
    sst = torch.sum((ynse - torch.mean(ynse))**2)
    nse = 1 - sse/sst
    test_loss /= num_batches
    print(f"Avg loss: {nse.item():>8f} \n")
    return nse.item(), pred_list, ynse

####################################################################################################
####################################################################################################
####################################################################################################

# read list of the basins which have undergone a regime change
direc = 'D:/Research/non_staitionarity/codes/results/data_split'
fname = 'set_A.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data  = fid.readlines()
fid.close()
basins = []
psd = []
ptrend = []
pdiff = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    basins.append(data_tmp[0])
    psd.append([float(x) for x in data_tmp[1:8]])
    ptrend.append([float(x) for x in data_tmp[8:15]])
    pdiff.append([float(x) for x in data_tmp[15:-2]])
psd = np.array(psd)
ptrend = np.array(ptrend)
pdiff = np.array(pdiff)

# keep the basins corresponding to statistically significant change in F4 (less than 2 weeks timescales component)
inds = np.nonzero((ptrend[:,4]<=0.05) & (pdiff[:,4]<=0.05))
inds = inds[0]
A = [basins[i] for i in inds]

# read the list of 531 basins
direc = 'D:/Research/non_staitionarity/data/531_basins'
fname = 'basin_list_zero_appended.txt'
filename = direc + '/' + fname
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
basins_531 = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    basins_531.append(data_tmp[0])

# remove the data from basins that are not contained in basins_531
A_final = []
for ind in range(0,len(A)):
    basin_tmp = A[ind]
    ii = [i for i in range(0, len(basins_531)) if basins_531[i] == basin_tmp]
    if len(ii) != 0:
        A_final.append(basin_tmp)
##################################################################################################
## read the list of basisn which have not undergone a regime change
# basins that are contained in 531 basin but not in set A_final or 'basins'
B = []
for basin_tmp in basins_531:
    if basin_tmp not in basins:
        B.append(basin_tmp)

########################################################################################

# read static data for all basins
direc_static = 'D:/Research/non_staitionarity/data/RF_static_data'
fname = 'RF_static.txt'
filename = direc_static + '/' + fname
fid = open(filename)
data = fid.readlines()
fid.close()
static_data = []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    static_data.append([data_tmp[0]] + [float(x) for x in data_tmp[1:]])
 
########################################################################################
# prepare data for LSTM regression
direc_dynamic = 'D:/Research/non_staitionarity/data/LSTM_dynamic_data'

train_datenum_1 = datetime.date(1980,10,1).toordinal()
train_datenum_2 = datetime.date(1989,9,30).toordinal()
val_datenum_1 = datetime.date(1989,10,1).toordinal()
val_datenum_2 = datetime.date(1994,9,30).toordinal()
test_datenum_1 = datetime.date(2001,10,1).toordinal()
#test_datenum_2 = datetime.date(1999,9,30).toordinal()

"""
ind_train_1 = 0
ind_train_2 = 3650
ind_val_1 = 3650
ind_val_2 = 5445
ind_test_1 = 5445
ind_test_2 = ''
"""

eps = 0.1 # to normalize MSE

train_data = []
val_data = []
test_data=[]
SD_y = []
test_data_A = []
basins_used = []
for basin in A_final + B:          ###########
    
    # read dynamic data
    fname = basin + '_met_dynamic.txt'
    filename = direc_dynamic + '/' + fname
    if os.path.exists(filename):
        
        basins_used.append(basin)
        met = readDynamic(filename)
    
        # join static and met data
        ind = [i for i in range(0, len(static_data)) if static_data[i][0] == basin]
        ind = ind[0]
        for wind in range(0,len(met)):
            met[wind] = met[wind] + static_data[ind][1:]
        met = np.array(met)
        
        # if dates are specified for training, validation, and testing period
        ind_train_1 = np.nonzero(met[:,0] == train_datenum_1)
        if len(ind_train_1[0]) != 0:
            ind_train_1 = ind_train_1[0][0]
        else:
            ind_train_1 = 0
        ind_train_2 = np.nonzero(met[:,0] == train_datenum_2)
        ind_train_2 = ind_train_2[0][0]
        ind_val_1 = np.nonzero(met[:,0] == val_datenum_1)
        ind_val_1 = ind_val_1[0][0]
        ind_val_2 = np.nonzero(met[:,0] == val_datenum_2)
        ind_val_2 = ind_val_2[0][0]
        ind_test_1 = np.nonzero(met[:,0] == test_datenum_1)
        ind_test_1 = ind_test_1[0][0]
        # ind_test_2 = np.nonzero(met[:,0] == test_datenum_2)
        # ind_test_2 = ind_test_2[0][0]

        train_tmp = met[ind_train_1:ind_train_2+1,:]
        val_tmp = met[ind_val_1:ind_val_2+1,:]
        test_tmp = met[ind_test_1:,:]

        # compute standard deviation of streamflow in training set
        sd_data = np.array(train_tmp)[:,1]
        sd = np.nanstd(sd_data) + eps
        SD_y.append(sd)

        # add standard deviation as the last predictor (it should be removed before model training)
        train_tmp = np.concatenate((train_tmp, sd*np.ones((train_tmp.shape[0], 1))), axis = 1)
        val_tmp = np.concatenate((val_tmp, sd*np.ones((val_tmp.shape[0], 1))), axis = 1)
        test_tmp = np.concatenate((test_tmp, sd*np.ones((test_tmp.shape[0], 1))), axis = 1)

        # remove NaNs
        train_tmp = removeNaN(train_tmp)
        val_tmp = removeNaN(val_tmp)
        test_tmp = removeNaN(test_tmp)
        
        train_data.append(train_tmp)
        val_data.append(val_tmp)
        test_data.append(test_tmp)
        if basin in A_final:
            test_data_A.append(test_tmp)

# finding mean and SD of predictor variables for normalization 
train_data_comb = np.concatenate(train_data)
mean_X = np.mean(train_data_comb[:,2:-1], axis = 0)
std_X = np.std(train_data_comb[:,2:-1], axis = 0)

# write mean_x and std_X data to a textfile
"""
normalization_stat = [mean_X,std_X]
direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global/' + save_subdir
filename  = direc + '/' + 'norm_stat.txt'
fid = open(filename, 'w')
fid.write('mean_X\tstd_X\n')
for wind in range(0,len(mean_X)):
    fid.write('%s\t%f\n'%(mean_X[wind], std_X[wind]))
fid.close()
"""

# convert data to torch tensors
train_ten = []
for train_tmp in train_data:
    train_ten.append(torch.from_numpy(train_tmp).float())

val_ten = []
for val_tmp in val_data:
    val_ten.append(torch.from_numpy(val_tmp).float())

test_ten = []
for test_tmp in test_data_A:            #################
    test_ten.append(torch.from_numpy(test_tmp).float())

mean_X = torch.from_numpy(mean_X).float()
std_X = torch.from_numpy(std_X).float()

# define dataloader
N = 2**8    # bacth size
train_dataset = CustomData(train_ten, 365, mean_X, std_X)
train_dataloader  = DataLoader(train_dataset, batch_size = N, shuffle=True, drop_last = True)
val_dataset = CustomData(val_ten, 365, mean_X, std_X)
val_dataloader  = DataLoader(val_dataset, batch_size = N, shuffle=True, drop_last = True)

######################################################################################################
######################################################################################################
# start model training (different models for different seeds)
seeds = [i for i in range(8)]
#seeds = [0]
test_obs_pred_seed = []
for seed in seeds:

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define LSTM model
    input_dim = len(mean_X)
    hidden_dim = 256
    n_layers = 1
    output_dim = 1
    lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
    lstm.cuda()

    # define loss and optimizer
    lossfn = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr = 10**(-3))

    # fix the number of epochs and start model training
    epochs = 50
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
    fname = 'model_exp2_' + str(seed) + '.pth'      ##########
    direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global/' + save_subdir
    filename = direc + '/' + fname
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

    # save NSE values to a textfile
    sname = 'basin_NSE_exp2_' + str(seed) + '.txt'      ##########
    save_direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global/' + save_subdir
    filename  = save_direc + '/' + sname
    fid = open(filename, 'w')
    fid.write('BASIN\tNSE\tsd\n')
    for wind in range(0,len(NSE)):
        fid.write('%s\t%f\t%f\n'%(basins_used[wind], NSE[wind], SD_y[wind]))
    fid.close()

    del lstm, optimizer, lossfn
###################################################################################################

# compute average prediction obtained from different seeds
obs_pred_avg  = []
for basin_ind in range(len(A_final)):
    ypred = 0
    for seed_ind in range(len(seeds)):
        ypred += test_obs_pred_seed[seed_ind][basin_ind][1]
    ypred /= len(seeds)
    yobs = test_obs_pred_seed[seed_ind][basin_ind][0]
    obs_pred_avg.append([yobs, ypred])
    
# compute NSE of average prediction
NSE_avg = []
for obs_pred in obs_pred_avg:
    nse = computeNSE(obs_pred[0], obs_pred[1])
    NSE_avg.append(nse)

# save NSE values to a textfile
sname = 'basin_NSE_exp2_' + 'avg' + '.txt'      #############
save_direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global/' + save_subdir
filename  = save_direc + '/' + sname
fid = open(filename, 'w')
fid.write('BASIN\tNSE\tsd\n')
for wind in range(0,len(NSE_avg)):
    fid.write('%s\t%f\t%f\n'%(basins_used[wind], NSE_avg[wind], SD_y[wind]))
fid.close()

# compute uncertainty in NSE
NSEBS = []
for obs_pred in obs_pred_avg:
    nse = computeNSEBS(obs_pred[0], obs_pred[1])
    NSEBS.append(nse)

# save NSE BS values
fname = 'nse_uncertainty_exp2'      ################
direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global/' + save_subdir
filename = direc + '/' + fname
pickle.dump(NSEBS, open(filename, 'wb'))

# load saved model
"""
fname = 'model_saved_normalized'
direc = 'D:/Research/non_staitionarity/codes/results/LSTM_global'
filename = direc + '/' + fname
state = torch.load(filename)
lstm.load_state_dict(state)
"""

a = 1