import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import csv
import math
import statistics
import gc

class CostNet(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims=1,
                 hidden_dims=2000,
                 prob=0.25,
                 use_cuda=True):
        super(CostNet, self).__init__()
        if use_cuda:
            self.cuda()
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_dims1 = math.floor(hidden_dims / 10)
        self.output_dims = output_dims
        
        # self.embedding = nn.EmbeddingBag(num_icd_codes, input_dims) # calculate word embeddings for icd codes?
        self.fc1 = nn.Linear(in_features=input_dims, out_features=hidden_dims, device=self.device)
        self.fc1bn = nn.BatchNorm1d(hidden_dims, device=self.device)
        self.fc2 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims, device=self.device)
        self.fc2bn = nn.BatchNorm1d(hidden_dims, device=self.device)
        self.fc3 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims, device=self.device)
        self.fc3bn = nn.BatchNorm1d(self.hidden_dims, device=self.device)
        self.fc4 = nn.Linear(in_features=hidden_dims, out_features=self.hidden_dims1, device=self.device)
        self.fc4bn = nn.BatchNorm1d(self.hidden_dims1, device=self.device)
        self.fc5 = nn.Linear(in_features=self.hidden_dims1, out_features=self.hidden_dims1, device=self.device)
        self.fc5bn = nn.BatchNorm1d(self.hidden_dims1, device=self.device)
        self.fc6 = nn.Linear(in_features=self.hidden_dims1, out_features=self.hidden_dims1, device=self.device)
        self.fc6bn = nn.BatchNorm1d(self.hidden_dims1, device=self.device)
        self.fc7 = nn.Linear(in_features=self.hidden_dims1, out_features=self.hidden_dims1, device=self.device)
        self.fc7bn = nn.BatchNorm1d(self.hidden_dims1, device=self.device)
        self.learntcost = nn.Linear(in_features=self.hidden_dims1, out_features=output_dims, device=self.device)
        self.dropout = nn.Dropout(prob)
    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = self.fc1bn(x)
        # x = self.dropout(x)
        x = F.selu(self.fc2(x))
        x = self.fc2bn(x)
        # x = self.dropout(x)
        x = F.selu(self.fc3(x))
        x = self.fc3bn(x)
        # x = self.dropout(x)
        x = F.selu(self.fc4(x))
        x = self.fc4bn(x)
        x = F.selu(self.fc5(x))
        x = self.fc5bn(x)
        cost = self.learntcost(x)
        return cost

# initialize for gc call
batch_features = 2
batch_targets = 1
features_test = 1
targets_test = 1
model = 1

for chunk in range(6): # 6 chunks - 0 through 5
    del(model, batch_features, batch_targets, features_test, targets_test)
    torch.cuda.empty_cache()
    model = CostNet(input_dims=data_x_train.shape[1], hidden_dims=350, prob = 0.25)
    
    data = pd.read_csv("HR_2019_split/HR_chunk" + str(chunk) + ".csv", header=0, dtype="float")
    # else:
    #     data = data.append(pd.read_csv("HR_2019_split/HR_chunk" + str(chunk) + ".csv", header=0, dtype="float"))
    colnames = list(data.columns)
    # prep data
    colnames = list(data.columns)
    colnames_other = colnames[0:66]
    colnames_icd = colnames[66:len(colnames)]
    colnames_output = ['died', 'los', 'totchg']

    ## binarize some variables
    data.loc[data["hcup_ed"] >= 1, "hcup_ed"] = 1
    data.loc[data["tran_in"] >= 1, "tran_in"] = 1
    data.loc[data["tran_out"] >= 1, "tran_out"] = 1
    # binarize h_contrl and race
    data.loc[data["h_contrl"] == 2, "h_contrl"] = -1
    data.loc[data["h_contrl"] > 0, "h_contrl"] = 1
    data.loc[data["race"] == 2, "race"] = -1
    data.loc[data["race"] == 3, "race"] = -1
    data.loc[data["race"] > 0, "race"] = 1

    ## convert binary 0,1 scale to -1,1 scale; standardize linear variables
    colnames_input_binary = colnames[0:20] + ['aweekend', 'elective', 'female', 'i10_multinjury', 'hcup_ed', 'tran_in', 'tran_out']
    colnames_input_linear = ['ccr_nis', 'wageindex', 'age', 'i10_injury', 'i10_ndx', 'i10_npr', 'los']
    colnames_input_linear_meanonly = ['amonth', 'discwt', 'zipinc_qrtl', 'aprdrg_risk_mortality', 'aprdrg_severity'] # + colnames_dx + colnames_pr # variance is so small for discwt and others are pseudo-categorical/ordered, that will try just centering mean=0
    colnames_output_binary = ['died']
    colnames_output_linear = ['totchg']

    num_other_inputs = len(colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly)

    # log(totchg) is normally distributed, log(los) appears somewhat normally distributed as well, but bound at 0
    # data = data.astype('float')
    ## scaling across whole dataset and storing in data_others
    ## data.loc[:, colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_output_binary + colnames_output_linear] = data_others.loc[(chunk * 1000000):((chunk+1)*1000000-1), colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_output_binary + colnames_output_linear]
   
    data.loc[:, 'los'] = np.add(data.loc[:, 'los'], 1)
    data.loc[:, colnames_output_linear] = np.log10(data.loc[:, colnames_output_linear])

    data.loc[:, colnames_output_binary] = np.multiply(np.add(data.loc[:, colnames_output_binary].values, -0.5), 2)
    data.loc[:, colnames_input_binary] = np.multiply(np.add(data.loc[:, colnames_input_binary].values, -0.5), 2)

    data.loc[:, colnames_output_linear] = scale(data.loc[:, colnames_output_linear], axis = 0, with_mean = True, with_std = True)
    data.loc[:, colnames_input_linear] = scale(data.loc[:, colnames_input_linear], axis = 0, with_mean = True, with_std = True)
    data.loc[:, colnames_input_linear_meanonly] = scale(data.loc[:, colnames_input_linear_meanonly], axis = 0, with_mean = True, with_std = False)

     # data = data.drop(['ccr_nis', 'wageindex', 'discwt'])

    ## choose output and prepare data for CUDA
    colnames_output = ['totchg']   

    data_x = data.loc[:, ['h_contrl', 'race'] + colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_icd]
    data_y = data.loc[:, colnames_output]

    data_x = data_x.values.astype(np.float32)
    data_y = data_y.values.astype(np.float32)

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 47523)

    features_train = torch.from_numpy(data_x_train)#.to(model.device)
    targets_train = torch.from_numpy(data_y_train)#.to(model.device)
    features_test = torch.from_numpy(data_x_test)#.to(model.device)
    targets_test = torch.from_numpy(data_y_test)#.to(model.device)
    model = model.to(model.device)

    batch_size = 100000 # data_x_test.shape[0]
    n_iters = 1000
    num_epochs = int(n_iters / (len(features_train) / batch_size))
    num_epochs = 61

    train = TensorDataset(features_train, targets_train)
    test = TensorDataset(features_test, targets_test)

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size*2, shuffle = False)

    loss_func = nn.MSELoss()
    lr = .01 # 
    wd = .0001 # 
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)

    for epoch in range(num_epochs):
        for i, (batch_features, batch_targets) in enumerate(train_loader):
            batch_features = batch_features.to(model.device)
            batch_targets = batch_targets.to(model.device)
            batch_features = Variable(batch_features)
            batch_targets = Variable(batch_targets)
            torch.cuda.empty_cache()
            prediction = model(batch_features)
            loss = loss_func(prediction, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if epoch % 10 == 0:
            print(epoch // 10, "train", loss.item())
            for features_test, targets_test in test_loader:
                features_test = features_test.to(model.device)
                targets_test = targets_test.to(model.device)
                torch.cuda.empty_cache()
                outputs = model(features_test)
                loss_test = loss_func(outputs, targets_test)
                print(epoch // 10, "test", loss_test.item())
                print(epoch // 10, "diff", (loss_test.item() - loss.item()))  
    
    print(str(chunk) + " done")

### rerun training without h_contrl or race
for chunk in [0, 1, 2, 3]: # range(6): # 6 chunks - 0 through 5
    del(model, batch_features, batch_targets, features_test, targets_test)
    torch.cuda.empty_cache()
    
    data = pd.read_csv("HR_2019_split/HR_chunk" + str(chunk) + ".csv", header=0, dtype="float")
    # else:
    #     data = data.append(pd.read_csv("HR_2019_split/HR_chunk" + str(chunk) + ".csv", header=0, dtype="float"))
    colnames = list(data.columns)
    # prep data
    colnames = list(data.columns)
    colnames_other = colnames[0:66]
    colnames_icd = colnames[66:len(colnames)]
    colnames_output = ['died', 'los', 'totchg']

    ## binarize some variables
    data.loc[data["hcup_ed"] >= 1, "hcup_ed"] = 1
    data.loc[data["tran_in"] >= 1, "tran_in"] = 1
    data.loc[data["tran_out"] >= 1, "tran_out"] = 1
    # binarize h_contrl and race
    data.loc[data["h_contrl"] == 2, "h_contrl"] = -1
    data.loc[data["h_contrl"] > 0, "h_contrl"] = 1
    data.loc[data["race"] == 2, "race"] = -1
    data.loc[data["race"] == 3, "race"] = -1
    data.loc[data["race"] > 0, "race"] = 1

    ## convert binary 0,1 scale to -1,1 scale; standardize linear variables
    colnames_input_binary = colnames[0:20] + ['aweekend', 'elective', 'female', 'i10_multinjury', 'hcup_ed', 'tran_in', 'tran_out']
    colnames_input_linear = ['ccr_nis', 'wageindex', 'age', 'i10_injury', 'i10_ndx', 'i10_npr', 'los']
    colnames_input_linear_meanonly = ['amonth', 'discwt', 'zipinc_qrtl', 'aprdrg_risk_mortality', 'aprdrg_severity'] # + colnames_dx + colnames_pr # variance is so small for discwt and others are pseudo-categorical/ordered, that will try just centering mean=0
    colnames_output_binary = ['died']
    colnames_output_linear = ['totchg']

    num_other_inputs = len(colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly)

    # log(totchg) is normally distributed, log(los) appears somewhat normally distributed as well, but bound at 0
    # data = data.astype('float')
    ## scaling across whole dataset and storing in data_others
    ## data.loc[:, colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_output_binary + colnames_output_linear] = data_others.loc[(chunk * 1000000):((chunk+1)*1000000-1), colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_output_binary + colnames_output_linear]
   
    data.loc[:, 'los'] = np.add(data.loc[:, 'los'], 1)
    data.loc[:, colnames_output_linear] = np.log10(data.loc[:, colnames_output_linear])

    data.loc[:, colnames_output_binary] = np.multiply(np.add(data.loc[:, colnames_output_binary].values, -0.5), 2)
    data.loc[:, colnames_input_binary] = np.multiply(np.add(data.loc[:, colnames_input_binary].values, -0.5), 2)

    data.loc[:, colnames_output_linear] = scale(data.loc[:, colnames_output_linear], axis = 0, with_mean = True, with_std = True)
    data.loc[:, colnames_input_linear] = scale(data.loc[:, colnames_input_linear], axis = 0, with_mean = True, with_std = True)
    data.loc[:, colnames_input_linear_meanonly] = scale(data.loc[:, colnames_input_linear_meanonly], axis = 0, with_mean = True, with_std = False)

     # data = data.drop(['ccr_nis', 'wageindex', 'discwt'])

    ## choose output and prepare data for CUDA
    colnames_output = ['totchg']   

    data_x = data.loc[:, colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_icd]
    data_y = data.loc[:, colnames_output]

    data_x = data_x.values.astype(np.float32)
    data_y = data_y.values.astype(np.float32)

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 47523)

    features_train = torch.from_numpy(data_x_train)#.to(model.device)
    targets_train = torch.from_numpy(data_y_train)#.to(model.device)
    features_test = torch.from_numpy(data_x_test)#.to(model.device)
    targets_test = torch.from_numpy(data_y_test)#.to(model.device)
    
    batch_size = 100000 # data_x_test.shape[0]
    n_iters = 1000
    num_epochs = int(n_iters / (len(features_train) / batch_size))
    num_epochs = 51

    train = TensorDataset(features_train, targets_train)
    test = TensorDataset(features_test, targets_test)

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

    model = CostNet(input_dims=data_x_train.shape[1], hidden_dims=350, prob = 0.25)
    model = model.to(model.device)
    loss_func = nn.MSELoss()
    lr = .01 # 
    wd = .0001 # 
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)

    for epoch in range(num_epochs):
        for i, (batch_features, batch_targets) in enumerate(train_loader):
            batch_features = batch_features.to(model.device)
            batch_targets = batch_targets.to(model.device)
            batch_features = Variable(batch_features)
            batch_targets = Variable(batch_targets)
            torch.cuda.empty_cache()
            prediction = model(batch_features)
            loss = loss_func(prediction, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if epoch % 10 == 0:
            print(epoch // 10, "train", loss.item())
            for features_test, targets_test in test_loader:
                features_test = features_test.to(model.device)
                targets_test = targets_test.to(model.device)
                torch.cuda.empty_cache()
                outputs = model(features_test)
                loss_test = loss_func(outputs, targets_test)
                print(epoch // 10, "test", loss_test.item())
                print(epoch // 10, "diff", (loss_test.item() - loss.item()))  
    
    print(str(chunk) + " done")

### rerun training without variables expected to not have predictive utility
for chunk in [0, 1, 2, 3]: # range(6): # 6 chunks - 0 through 5
    del(model, batch_features, batch_targets, features_test, targets_test)
    torch.cuda.empty_cache()
    
    data = pd.read_csv("HR_2019_split/HR_chunk" + str(chunk) + ".csv", header=0, dtype="float")
    # else:
    #     data = data.append(pd.read_csv("HR_2019_split/HR_chunk" + str(chunk) + ".csv", header=0, dtype="float"))
    colnames = list(data.columns)
    # prep data
    colnames = list(data.columns)
    colnames_other = colnames[0:66]
    colnames_icd = colnames[66:len(colnames)]
    colnames_output = ['died', 'los', 'totchg']

    ## binarize some variables
    data.loc[data["hcup_ed"] >= 1, "hcup_ed"] = 1
    data.loc[data["tran_in"] >= 1, "tran_in"] = 1
    data.loc[data["tran_out"] >= 1, "tran_out"] = 1
    # binarize h_contrl and race
    data.loc[data["h_contrl"] == 2, "h_contrl"] = -1
    data.loc[data["h_contrl"] > 0, "h_contrl"] = 1
    data.loc[data["race"] == 2, "race"] = -1
    data.loc[data["race"] == 3, "race"] = -1
    data.loc[data["race"] > 0, "race"] = 1

    ## convert binary 0,1 scale to -1,1 scale; standardize linear variables
    colnames_input_binary = colnames[0:20] + ['aweekend', 'elective', 'female', 'i10_multinjury', 'hcup_ed', 'tran_in', 'tran_out']
    colnames_input_linear = ['ccr_nis', 'wageindex', 'age', 'i10_injury', 'i10_ndx', 'i10_npr']
    colnames_input_linear_meanonly = ['amonth', 'discwt', 'zipinc_qrtl', 'aprdrg_risk_mortality', 'aprdrg_severity'] # + colnames_dx + colnames_pr # variance is so small for discwt and others are pseudo-categorical/ordered, that will try just centering mean=0
    colnames_output_binary = ['died']
    colnames_output_linear = ['totchg']

    num_other_inputs = len(colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly)

    # log(totchg) is normally distributed, log(los) appears somewhat normally distributed as well, but bound at 0
    # data = data.astype('float')
    ## scaling across whole dataset and storing in data_others
    ## data.loc[:, colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_output_binary + colnames_output_linear] = data_others.loc[(chunk * 1000000):((chunk+1)*1000000-1), colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_output_binary + colnames_output_linear]
   
    data.loc[:, 'los'] = np.add(data.loc[:, 'los'], 1)
    data.loc[:, colnames_output_linear] = np.log10(data.loc[:, colnames_output_linear])

    data.loc[:, colnames_output_binary] = np.multiply(np.add(data.loc[:, colnames_output_binary].values, -0.5), 2)
    data.loc[:, colnames_input_binary] = np.multiply(np.add(data.loc[:, colnames_input_binary].values, -0.5), 2)

    data.loc[:, colnames_output_linear] = scale(data.loc[:, colnames_output_linear], axis = 0, with_mean = True, with_std = True)
    data.loc[:, colnames_input_linear] = scale(data.loc[:, colnames_input_linear], axis = 0, with_mean = True, with_std = True)
    data.loc[:, colnames_input_linear_meanonly] = scale(data.loc[:, colnames_input_linear_meanonly], axis = 0, with_mean = True, with_std = False)

     # data = data.drop(['ccr_nis', 'wageindex', 'discwt'])

    ## choose output and prepare data for CUDA
    colnames_output = ['totchg']   

    data_x = data.loc[:, ['h_contrl', 'race'] + colnames_input_binary + colnames_input_linear + colnames_input_linear_meanonly + colnames_icd]
    data_y = data.loc[:, colnames_output]

    data_x = data_x.values.astype(np.float32)
    data_y = data_y.values.astype(np.float32)

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 47523)

    features_train = torch.from_numpy(data_x_train)#.to(model.device)
    targets_train = torch.from_numpy(data_y_train)#.to(model.device)
    features_test = torch.from_numpy(data_x_test)#.to(model.device)
    targets_test = torch.from_numpy(data_y_test)#.to(model.device)
    
    batch_size = 100000 # data_x_test.shape[0]
    n_iters = 1000
    num_epochs = int(n_iters / (len(features_train) / batch_size))
    num_epochs = 51

    train = TensorDataset(features_train, targets_train)
    test = TensorDataset(features_test, targets_test)

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

    model = CostNet(input_dims=data_x_train.shape[1], hidden_dims=350, prob = 0.25)
    model = model.to(model.device)
    loss_func = nn.MSELoss()
    lr = .01 # 
    wd = .0001 # 
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)

    for epoch in range(num_epochs):
        for i, (batch_features, batch_targets) in enumerate(train_loader):
            batch_features = batch_features.to(model.device)
            batch_targets = batch_targets.to(model.device)
            batch_features = Variable(batch_features)
            batch_targets = Variable(batch_targets)
            torch.cuda.empty_cache()
            prediction = model(batch_features)
            loss = loss_func(prediction, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if epoch % 10 == 0:
            print(epoch // 10, "train", loss.item())
            for features_test, targets_test in test_loader:
                features_test = features_test.to(model.device)
                targets_test = targets_test.to(model.device)
                torch.cuda.empty_cache()
                outputs = model(features_test)
                loss_test = loss_func(outputs, targets_test)
                print(epoch // 10, "test", loss_test.item())
                print(epoch // 10, "diff", (loss_test.item() - loss.item()))  
    
    print(str(chunk) + " done")



















