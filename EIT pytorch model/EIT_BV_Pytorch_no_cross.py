import time
import datetime
import os
import random
import pandas as pd
import numpy as np
from numpy import linalg as LA
import pickle
import matplotlib.pyplot as plt
import itertools
np.random.seed(1337) # for reproducibility

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from torchsummary import summary

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score #for manual measurement of accuracy using predicted labels
from sklearn.metrics import confusion_matrix



# Alternative order / original FMG before smoothing Or smoothed FMG , according to the selected path
data = []
data = np.zeros((800, 40+1))

r = 0  # r is row of the data matrix
for g in range(0, 10):   # 0<=i>10 # 10 gestures
    for p in range(1, 9):  #8 persons
        df = pd.read_excel(r'/home/maroueneubuntu/Desktop/CNN/EIT Boundary Voltage/EIT Boundary Voltage/p'+str(p)+'/BV'+str(p)+'_'+str(g)+'.xlsx', header=None)
        df = df.values

        for i in range(0, 10):  #trial 1-10
            data[r, 0] = g
            data[r, 1:] = df[:, i+1]/LA.norm(df[:, i+1])  #first coloumn in the BV excel sheet is refernce
            r = r + 1
# splitting into train and testing 1st 8 for training, last 2 trials testing
X, Ytr, Xtest, Yts = [], [], [], []
for i in range(0, len(data), 10):  # each 10 steps has the same gesture and for the same subject
    # training and val set
    for j in range(0, 8):  # from 0 to 7 (8 indices)
        X.append(data[i + j, 1:])
        Ytr.append(data[i + j, 0])
    # testing set
    c = 8
    for j in range(0, 2):
        Xtest.append(data[i + c, 1:])
        Yts.append(data[i + c, 0])
        c = c + 1

print('length of training data: ', len(X))
print('length of the training labels: ', len(Ytr))
print('length of the testing data: ', len(Xtest))
print('length of the testing labels: ', len(Yts))

X_fmg_train = np.array(X)
X_fmg_train = X_fmg_train.reshape(len(X_fmg_train), X_fmg_train.shape[1], 1)  #  640x40x1 len(X_fmg_train) or X_fmg_train.shape[0]
X_fmg_test = np.array(Xtest)
X_fmg_test = X_fmg_test.reshape(len(X_fmg_test), X_fmg_test.shape[1], 1)   #160x40x1

class MyDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms=transforms

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # if self.transforms:
        #     item = self.transforms(X[index])
        return torch.transpose(torch.tensor(self.X[index],dtype=torch.float32),0,1), int(self.Y[index])

    def __len__(self):
        return len(self.X)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,25, kernel_size=(3))
        self.bn1 = nn.BatchNorm1d(25)
        self.pool1 = nn.MaxPool1d(kernel_size=3,stride=2,padding=(3 - 1) // 2 )
        self.drop1=nn.Dropout(0.2)
        self.fc1=nn.Linear(475,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
        self.flatten=nn.Flatten()

    def forward(self,x):
        # print(x.size())
        x=(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.bn1(x)
        # print(x.size())
        x=self.pool1(x)
        # print(x.size())
        x=self.drop1(x)
        # print(x.size())
        # x=torch.flatten(x)
        # x=x.view(32,-1)
        x=self.flatten(x)
        # print(x.size())
        x=F.relu(self.fc1(x))
        # print(x.size())
        x=F.relu(self.fc2(x))
        # print(x.size())
        x=self.fc3(x)
        # print(x.size())
        return x


# hyper-params
n_epochs=250
batch_size=32
learning_rate=0.001

# datasets
transform=transforms.Compose([transforms.ToTensor()])
train_dataset=MyDataset(X=X_fmg_train,Y=Ytr,transforms=transform)
test_dataset=MyDataset(X=X_fmg_test,Y=Yts,transforms=transform)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
classes=('0','1','2','3','4','5','6','7','8','9')

# print(train_dataset[639])
# print(test_dataset[17])
print('length of training data:',len(train_dataset))
print('length of test:',len(test_dataset))
print('shape of input',train_dataset[0][0].size())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model=Net().to(device)
crieterion=nn.CrossEntropyLoss()
# opt=torch.optim.SGD(model.parameters(),lr=learning_rate)
opt=torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps=len(train_loader)
a=0
b=0
for epoch in range(n_epochs):
    for i,(images,labels) in enumerate(train_loader,0):
        # for j in range(len(labels)):
        #     labels[j] = random.randint(0, 9)
        # # print('ok')
        if a==0:
            
            print(labels)
            print (images[0].size(),labels[0])
            a+=1
        b+=1
        # forward pass
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=crieterion(outputs,labels)

        # backward and upadte
        opt.zero_grad()
        loss.backward()
        opt.step()
        # print (i)
        if(i+1)%20==0:
            print(f'epoch[{epoch+1}/{n_epochs}], step [{i+1}/{n_total_steps}], loss{loss.item():.4f}')
print(b)
print('Finished training')
print(model)
summary(model, (1, 40), batch_size=32,device="cpu")
with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_samples=[0 for i in range(10)]

    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        n_samples+=predicted.size(0)
        n_correct+=(predicted==labels).sum().item() 

        for i in range(batch_size):
            label=labels[i]
            pred=predicted[i]
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    acc=100.0*n_correct/n_samples
    print(f'accuracy of the network ={acc:.4f}%')

    for i in range(10):
        acc=100*n_class_correct[i]/n_class_samples[i]
        print(f'accurcay of class {classes[i]}= {acc:.4f}%')