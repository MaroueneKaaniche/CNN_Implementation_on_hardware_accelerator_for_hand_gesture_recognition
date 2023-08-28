import numpy as np
import pandas as pd
from numpy import linalg as LA
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import ai8x


def GetXY_train_test():
    """
    return X train and test with their according labels Y in the following order
    X_train,Y_train,X_test,Y_test
    """
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
    return X_fmg_train,Ytr,X_fmg_test,Yts

class CustomDataset(Dataset):
    def __init__(self,args, root_dir, d_type,X_arg,Y_arg,transform=None):
        self.d_type=d_type
        self.transform=transform
        self.root_dir=root_dir
        self.args=args
        self.X=X_arg.copy()
        self.Y=Y_arg.copy()
        if self.args.act_mode_8bit:
            self.X[:,:]=np.round((self.X[:,:] - 0.5) * 256).clip(-128, 127)
        else:
            self.X[:,:]= (np.round((self.X[:,:] - 0.5) * 256).clip(-128, 127))/(128.)
        # self.X=(self.X.astype(np.int32))
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # if self.transforms:
        #     item = self.transforms(X[index])
        return torch.transpose(torch.FloatTensor(self.X[index]),0,1), np.compat.long(self.Y[index])

    def __len__(self):
        return len(self.X)
         
def CustomDataset_get_datasets(data,load_train=True,load_test=True):
    
    (data_dir,args) = data
    Xtr,Ytr,Xtest,Ytest=GetXY_train_test()
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor()])
        train_dataset=CustomDataset(args,root_dir=data_dir, d_type='train',X_arg=Xtr,Y_arg=Ytr,transform=train_transform)
        print(f'Train dataset length: {len(train_dataset)}\n')
        print(f'some training samples: {train_dataset[0]}')
    else:
        train_dataset=None
    
    if load_test:
        test_transform = transforms.Compose([transforms.ToTensor()])
        test_dataset=CustomDataset(args,root_dir=data_dir,d_type='test',X_arg=Xtest, Y_arg=Ytest,transform=test_transform)
        print(f'Test dataset length: {len(test_dataset)}\n')
        print(f'some test samples: {test_dataset[0]}')
    else:
        test_dataset=None
    
    return train_dataset, test_dataset
          
datasets=[
    {
    'name':'CustomDataset2',
    'input':(1,40),
    'output':(0,1,2,3,4,5,6,7,8,9),
    'loader': CustomDataset_get_datasets,
    }

]

# ---------------------- testing class and dataloader ----------------------

# a,b,c,d=GetXY_train_test()
# class argz:
#     def __init__(self):
#         self.act_mode_8bit=True
# testargz=argz()

# dataset=CustomDataset(testargz,'aaa','Train',a,b)
# dataset2=CustomDataset(testargz,'aaa','Train',c,d)
# print(dataset[638])
# print(dataset[120])
# print(dataset[502])

# print(len(dataset))
# print(dataset2[124])
# print(dataset2[23])
# print(dataset2[54])
# print(len(dataset2))

# directory_for_test="ahla"
# argument_for_test=(directory_for_test,testargz)
# train,test=CustomDataset_get_datasets(argument_for_test,load_train=True,load_test=True)
# print(train[638])
# print(train[120])
# print(train[502])
# print(test[124])
# print(test[23])
# print(test[54])