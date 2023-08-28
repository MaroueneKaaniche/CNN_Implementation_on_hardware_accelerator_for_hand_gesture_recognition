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

from sklearn.model_selection import KFold

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
        x=self.bn1(x)
        # print(x.size())
        x=self.pool1(x)
        # print(x.size())
        x=self.drop1(x)
        # x=torch.flatten(x)
        # x=x.view(32,-1)
        x=self.flatten(x)
        # print(x.size())
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# hyper-params
k_folds = 5
n_epochs=250
batch_size=32
learning_rate=0.001

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

results_acc = []
results_loss = []
Early_stopping_ids=[]

# datasets
transform=transforms.Compose([transforms.ToTensor()])
train_dataset=MyDataset(X=X_fmg_train,Y=Ytr,transforms=transform)
test_dataset=MyDataset(X=X_fmg_test,Y=Yts,transforms=transform)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
classes=('0','1','2','3','4','5','6','7','8','9')

# print(train_dataset[639])
# print(test_dataset[17])
# kfold = KFold(n_splits=k_folds, shuffle=True,random_state=123)
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True,random_state=123)

print('length of training data:',len(train_dataset))
print('length of test:',len(test_dataset))
print('shape of input',train_dataset[0][0].size())
model=Net()

for fold, (train_ids, test_ids) in enumerate(kfold.split(X_fmg_train, Ytr)):
    random.seed(4)
    random.shuffle(train_ids)
    random.shuffle(test_ids)
    print(f'FOLD {fold}')
    print('--------------------------------')
    crieterion=nn.CrossEntropyLoss()
    # opt=torch.optim.SGD(model.parameters(),lr=learning_rate)
    opt=torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps=len(train_loader)
    a=0
    b=0
    # print('--------------------------------')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    trainloader = torch.utils.data.DataLoader(
                      train_dataset, 
                      batch_size=32, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      train_dataset,
                      batch_size=32, sampler=test_subsampler)
    # network = Net()
    # network.apply(reset_weights)
    # reset_weights(m=model)
    correct,total=0,0
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(n_epochs):
        train_loss = 0
        train_acc = 0
        total_train = 0
        correct_train = 0
        model.train()
        for i,(images,labels) in enumerate(trainloader):
            if a==0:
                
                print (images[0].size(),labels[0])
                a+=1
            b+=1
            # forward pass
            outputs=model(images)
            loss=crieterion(outputs,labels)

            # backward and upadte
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_loss += loss.item()

        # training accuracy and loss
        train_loss /= len(trainloader.dataset)
        train_acc = 100 * correct_train / total_train
        # Validate the model
        total_loss = 0
        total_correct = 0
        total_examples = 0        
        model.eval()
        with torch.no_grad():
        # Iterate over the test data and generate predictions
            for i, (inputs, targets) in enumerate(testloader):

            # Generate outputs
                outputs = model(inputs)
                loss = crieterion(outputs, targets)
                total_loss += loss.item() * len(targets)
            # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                total_examples=total
                total_correct=correct
            val_loss = total_loss / total_examples
            val_acc = 100* (total_correct / total_examples)
        print(f"Epoch {epoch + 1} - train_loss:{train_loss:.4f}, val_loss: {val_loss:.4f}, train_acc: {train_acc:.4f} , val_acc: {val_acc:.4f}")
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            counter=0
        else:
            counter+=1
            if counter==patience:
                print('Early stopping after {} epochs'.format(epoch+1))
                Early_stopping_ids.append(epoch+1)
                break

    # # Print accuracy for fold
    # print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
    # print('--------------------------------')
    # results[fold] = 100.0 * (correct / total)        





    print(b)
    print(f'Finished training fold number{fold}')
    print(model)
    print('>>>>---------- Model summary ----------<<<<')
    # summary(model, (1, 40), batch_size=32,device="cpu")
    summary(model, (1, 40), batch_size=32,device="cpu")
    print('test accuracy for fold n',fold)
    with torch.no_grad():
        test_loss=0
        n_correct=0
        n_samples=0
        n_class_correct=[0 for i in range(10)]
        n_class_samples=[0 for i in range(10)]

        for images,labels in test_loader:
            images=images
            labels=labels
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            n_samples+=predicted.size(0)
            n_correct+=(predicted==labels).sum().item() 
            loss2=crieterion(outputs,labels)
            for i in range(batch_size):
                label=labels[i]
                pred=predicted[i]
                if (label==pred):
                    n_class_correct[label]+=1
                n_class_samples[label]+=1
            test_loss += loss2.item()
        acc=100.0*n_correct/n_samples
        network_acc=acc
        print(f'accuracy of the network ={network_acc:.4f}%')
        test_loss /= len(test_loader.dataset)
        print (f'loss of the network={test_loss:.4f}')
        for i in range(10):
            acc=100*n_class_correct[i]/n_class_samples[i]
            print(f'accurcay of class {classes[i]}= {acc:.4f}%')
    # Print accuracy for fold
    results_acc.append(network_acc)
    results_loss.append(test_loss)
        
print('------------------------------------------------------------------------')
print(' >>>> Scores per fold <<<<')
for i in range(0, len(results_acc)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {round(results_loss[i],3)} - Accuracy: {round(results_acc[i],2)}%')

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {round(np.mean(results_acc),2)} (+- {round(np.std(results_acc),2)})')
print(f'> Loss: {round(np.mean(results_loss),3)}')
print('------------------------------------------------------------------------')

print('------------------------------------------------------------------------')
print(' >>>> Early stopping indexes per fold <<<<')
for i in range(0, len(Early_stopping_ids)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} stopped at epoch {(Early_stopping_ids[i])}')

print(results_acc)