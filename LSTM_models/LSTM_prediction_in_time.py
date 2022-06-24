"""
LSTM regression model for one watershed

Author: Abhinav Gupta (Created: 19 Apr 2022)

"""
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

import torch
from torch import classes, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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

######################################################################################################
# read all the rain-dominated watersheds data
direc = 'D:/Research/non_staitionarity/data/LSTM_dynamic_data'
listFiles = os.listdir(direc)
basins = []

for fname in listFiles:
    basins.append(fname.split('_')[0])

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

########################################################################################################
# LSTM classes
########################################################################################################
# build a dataset class
class CustomData(Dataset):
    def __init__(self, X, y, seq_len):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y
        self.L = seq_len    # sequence length
    
    def __len__(self):
        return len(self.y) - self.L
    
    def __getitem__(self, index):
        x1 = self.X[index : index + self.L, :]
        y1 = self.y[index + self.L - 1]
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
        self.dropout = nn.Dropout(p = 0.30)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #out = self.fc(out[:,-1,:]) #use hn instead, use relu before fc
        # out = self.fc(self.dropout(hn[0,:,:]))
        out = self.fc(hn[0,:,:])
        #out=self.fc1(out)
        return out

# define the module to train the model
def train_mod(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    ynse_train = []
    pred_train = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return model.state_dict()

# define the module to validate the model
def validate_mod(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# define the module to test the model
def test_mod(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    sse = 0
    ynse = []
    pred_list = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            sse += torch.sum((pred - y)**2)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            ynse.append(y)
            pred_list.append(pred)
    ynse = torch.cat(ynse)
    pred_list = torch.cat(pred_list)
    sst = torch.sum((ynse - torch.mean(ynse))**2)
    nse = 1 - sse/sst
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {nse.item():>8f} \n")
    return nse.item(), pred_list, ynse

####################################################################################################
####################################################################################################
####################################################################################################


# prepare data for LSTM regression
direc_dynamic = 'D:/Research/non_staitionarity/data/LSTM_dynamic_data'
direc_save = 'D:/Research/non_staitionarity/codes/results/LSTM_prediction_in_time'

# prepare training data
train_data = []
for basin in basins:

    # read dynamic data
    fname = basin + '_met_dynamic.txt'
    filename = direc_dynamic + '/' + fname
    met_data = readDynamic(filename)

    # read static data
    ind = [i for i in range(0,len(static_data)) if static_data[i][0]==basin]
    ind = ind[0]
    static_tmp = static_data[ind][1:]
    area = static_tmp[16]
   
    train = np.array(met_data)

    # train set
    xtrain = train[0:3650,2:]
    ytrain = 24*3.6*0.02832*train[0:3650,1]/area

    # validation set
    xval = train[3650:5445,2:]
    yval = 24*3.6*0.02832*train[3650:5445,1]/area

    # test set
    xtest = train[5445:,2:]
    ytest = 24*3.6*0.02832*train[5445:,1]/area

    # remove rows containing NaNs from train data
    nanind = (np.isnan(xtrain).any(axis=1))
    nanind = np.nonzero(nanind==True)
    nanind = nanind[0]
    xtrain = np.delete(xtrain, tuple(nanind), axis=0)
    ytrain = np.delete(ytrain, tuple(nanind), axis=0)

    # remove rows containing NaNs from test data
    nanind = (np.isnan(xtest).any(axis=1))
    nanind = np.nonzero(nanind==True)
    nanind = nanind[0]
    xtest = np.delete(xtest, tuple(nanind), axis=0)
    ytest = np.delete(ytest, tuple(nanind), axis=0)

    # remove rows containing NaNs from val data
    nanind = (np.isnan(xval).any(axis=1))
    nanind = np.nonzero(nanind==True)
    nanind = nanind[0]
    xval = np.delete(xval, tuple(nanind), axis=0)
    yval = np.delete(yval, tuple(nanind), axis=0)

    # convert data to tensor
    xtrain = torch.from_numpy(xtrain).float()
    ytrain = torch.from_numpy(ytrain.reshape(1,-1).T).float()
    xval = torch.from_numpy(xval).float()
    yval = torch.from_numpy(yval.reshape(1,-1).T).float()
    xtest = torch.from_numpy(xtest).float()
    ytest = torch.from_numpy(ytest.reshape(1,-1).T).float()

    # normalize the data
    xmax, _ = torch.max(xtrain, 0)
    xtrain = torch.div(xtrain, xmax)
    xval = torch.div(xval, xmax)
    xtest = torch.div(xtest, xmax)

    # create data loaders
    N = 2**8     # batch size
    train_dataset = CustomData(xtrain, ytrain, 60)
    val_dataset = CustomData(xval, yval, 60)
    test_dataset = CustomData(xtest, ytest, 60)
    train_dataloader = DataLoader(train_dataset, batch_size = N, shuffle=True, drop_last = True)
    val_dataloader = DataLoader(val_dataset, batch_size = val_dataset.__len__())
    
    g = torch.Generator()
    g.manual_seed(0)
    test_dataloader = DataLoader(test_dataset, batch_size = test_dataset.__len__(), generator=g)

    # Execute modeling for 8 random seeds 
    nse_ts_list = []
    ypred_list = []
    yobs_list = []
    for seed in range(8):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # instantiate the model class
        input_dim = xtrain.shape[1]
        hidden_dim = 100
        n_layers = 1
        output_dim = 1
        lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
        lstm.cuda()

        # define loss and optimizer
        lossfn = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr = 10**(-2))

        # fix the number of epochs and start model training
        epochs = 100
        loss_vl_list = []
        model_state = []
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            state = train_mod(train_dataloader, lstm, lossfn, optimizer)
            loss_vl = validate_mod(val_dataloader, lstm, lossfn)
            model_state.append(copy.deepcopy(state))
            loss_vl_list.append(loss_vl)
        ind = np.nonzero(loss_vl_list == np.min(loss_vl_list))
        lstm.load_state_dict(model_state[ind[0][0]], strict = True)    
        nse_ts, ypred, yobs = test_mod(test_dataloader, lstm, lossfn)
        ypred = [ypred[ind][0].cpu().detach().numpy() for ind in range(len(ypred))]
        yobs = [yobs[ind][0].cpu().detach().numpy() for ind in range(len(yobs))]
        nse_ts_list.append(nse_ts)
        ypred_list.append(ypred)
        yobs_list.append(yobs)
        del lstm, optimizer, lossfn
 
    ypred = 0
    for y in ypred_list: ypred += np.array(y)
    ypred /= 8
    nse = computeNSE(yobs, ypred)
    a = 1

    # save test streamflow data to a textfile along with nse of prediction on test set
    sname = basin + '_test_obs_pred.txt'
    filename = direc_save + '/' + sname
    fid = open(filename, 'w')
    fid.write('NSE = ' + str(nse) + '\n')
    fid.write('Observed\tPredicted\n')
    for wind in range(0,len(yobs)):
        fid.write('%f\t%f\n'%(yobs[wind], ypred[wind]))
    fid.close()

    plt.scatter(yobs, ypred, s = 5)
    lim = np.max((np.max(yobs), np.max(ypred)))
    plt.ylim(0, lim)
    plt.xlim(0, lim)
    plt.plot([0, lim],[0, lim], color = 'black')
    plt.title('NSE = ' + str(round(nse*100)/100))
    plt.xlabel('Observed Streamflow (mm/day)')
    plt.ylabel('Estimated Streamflow (mm/day)')
    
    # save plot
    sname = basin + '_obs_vs_pred_plot.png'
    filename = direc_save + '/obs_vs_pred_plots/' + sname
    plt.savefig(filename, dpi = 300)
    plt.close()

    del train_dataset, val_dataset, test_dataset,  
    