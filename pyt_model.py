import ast
import numpy as np
import time
import sys
import pandas as pd
import subprocess
import pickle
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def define_model(input_size):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 32)
            self.fc4 = nn.Linear(32, 8)
            self.fc5 = nn.Linear(8, 2)
            #self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            #x = self.dropout(x)
            x = F.relu(self.fc2(x))
            #x = self.dropout(x)
            x = F.relu(self.fc3(x))
            #x = self.dropout(x)
            x = F.relu(self.fc4(x))
            #x = self.dropout(x)
            x = self.fc5(x)
            return x
        
        def predict(self,x):
            pred = F.softmax(self.forward(x), dim=1)
            ans = []
            for t in pred:
                if t[0]>t[1]:
                    ans.append(0)
                else:
                    ans.append(1)
            return torch.tensor(ans)
    return Net()

def train_model(model, n_epochs, loaders, mutation_name):
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    train_loader, valid_loader, test_loader = loaders

    valid_loss_min = np.Inf
    t_losses = []
    v_losses = []

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        # Training
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data.float())
            target = target.long()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
        # Validation
        model.eval()
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data.float())
            target = target.long()
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            
        # Saving loss for validation graph
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        t_losses.append(train_loss)
        v_losses.append(valid_loss)
        
        # save model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), 'model_files/model_{mutation_name}.pt')
            valid_loss_min = valid_loss
        
        # Progress bar
        print(f"\rEpoch: {epoch+1}\tTrain: {train_loss}\tVal: {valid_loss}\tMin Val: {valid_loss_min}", end='')
    print("")
    torch.save(model.state_dict(), 'model_files/model_final_{mutation_name}.pt')
    return t_losses, v_losses

def eval_model(mutation_name, state, test_loader, input_size):
    # evaluation of final model (after n epochs) or saved model with lowest validation loss
    model = define_model(input_size)
    if state == 'final':
        model.load_state_dict(torch.load('model_files/model_final_{mutation_name}.pt'))
    else:
        model.load_state_dict(torch.load('model_files/model_{mutation_name}.pt'))
    model.eval()
    score = 0
    try:
        for data, target in test_loader:
            data = data.to(device)
            data.float()
            pred = model.predict(data.float())
            pred.cpu()
            score += roc_auc_score(pred, target)*len(target)
            # print(pred)
            # print(target)
            # print(roc_auc_score(pred, target))
        print(f"\n{state}: ", score/len(test_loader.sampler))
    except:
        print("Predicted all 0...")