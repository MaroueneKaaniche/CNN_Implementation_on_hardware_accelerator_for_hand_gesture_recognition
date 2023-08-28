from tensorflow.keras import datasets
import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.model_selection import train_test_split



def customloader_get_atasets(data_dir):
    
    print('data directory', data_dir)
    (train_data, train_labels), (test_data,test_labels)=get_Xtrain_Xtest()
    
    # split to train, valid and test
    # train_data, valid_data, train_labels, valid_labels = train_test_split(
    #     train_data, train_labels, test_size=0.1, random_state=42)

    # changing data range
    train_data=normalize(train_data)
    # valid_data=normalize(valid_data)
    test_data=normalize(test_data)

    # make label shape: (n,)
    train_labels = train_labels.flatten()
    # valid_labels = valid_labels.flatten()
    test_labels = test_labels.flatten()

    return (train_data,train_labels), (test_data, test_labels)
    

def get_Xtrain_Xtest():
    """
    get data from the excel spreadsheets and split it between test and train data
    """
    data = []
    data = np.zeros((800 , 40+ 1))

    r = 0  # r is row of the data matrix
    for g in range(0, 10):   # 0<=i>10 # 10 gestures
        for p in range(1, 9):  #8 persons

            #df = pd.read_excel( r'D:\Thesis\EIT BV\EIT Boundary Voltage\p'+str(p)+'\BV'+str(p)+'_'+str(g)+'.xlsx', header=None)
            df = pd.read_excel( r'/home/maroueneubuntu/Desktop/CNN/EIT Boundary Voltage/EIT Boundary Voltage/p'+str(p)+'/BV'+str(p)+'_'+str(g)+'.xlsx', header=None)
            df = df.values

            for i in range(0, 10):  #trial 1-10
                data[r, 0] = g;
                data[r, 1:]=df[:,i+1]/LA.norm(df[:,i+1])  #first coloumn in the BV excel sheet is refernce
                r = r + 1;

    # print("data", type(data))
    # print('data shape: ', data.shape)
    # print(len(data))

    X, Ytr, Xtest, Yts = [], [], [], []
    for i in range(0, len(data), 10):  # each 10 steps has the same gesture and for the same subject
        # training and val set
        for j in range(0, 8):  # from 0 to 7 (8 indicies)
            X.append(data[i + j, 1:])
            Ytr.append(data[i + j, 0])
        # testing set
        c = 8
        for j in range(0, 2):
            Xtest.append(data[i + c, 1:])
            Yts.append(data[i + c, 0])
            c = c + 1

    # print('length of training data: ', len(X), type(X))
    # print('length of the training labels: ', len(Ytr), type(Ytr))
    # print('length of the testing data: ', len(Xtest), type(Xtest))
    # print('length of the testing labels: ', len(Yts), type(Yts))
    X_fmg_train = np.array(X)
    X_fmg_train = X_fmg_train.reshape(len(X_fmg_train), X_fmg_train.shape[1], 1)  # len(X_fmg_train) or X_fmg_train.shape[0]
    X_fmg_test = np.array(Xtest)
    X_fmg_test = X_fmg_test.reshape(len(X_fmg_test), X_fmg_test.shape[1], 1)
    Yts = np.array(Yts)
    Ytr = np.array(Ytr)    
    # print('Xtr type', type(X_fmg_train))    
    # print('X train shape',X_fmg_train.shape)
    # print('--------')
    # print('Y train shape',Ytr.shape)
    # print('Ytr type', type(Ytr))
    # print('shape is:' ,X_fmg_train[1], (X_fmg_train.shape[1], 1))
    return (X_fmg_train,Ytr), (X_fmg_test,Yts)  


def normalize(data):
    """
    scale data to the expected data range ; [-128,127]
    """ 
    data=np.round((data - 0.5) * 256).clip(-128, 127)
    data=(data.astype(np.int32))
    return data

def get_datasets(data_dir):
    """
    generic get the dataset in form of (train_data,train_labels), (valid_data, valid_labels),
    (test_data, test_labels)
    """
    return customloader_get_atasets(data_dir)

def get_classnames():
    """
    name of labels
    """
    class_names = list(map(str, range(10)))
    return class_names


# a=get_classnames()
# print(a)
# print('done loading data')
# get_Xtrain_Xtest()

# (a,b),(c,d)=get_Xtrain_Xtest()
# a=np.round((a - 0.5) * 256).clip(-128, 127)
# a=(a.astype(np.int32))
# print('------------------------')
# print('------------------------')
# print('------------------------')
# print(a.shape)
# print(a[len(a)-1])
# (a,b), (c,d), (e,f)=customloader_get_atasets()
# print("length of a", len(a),type(a), a.shape)
# print("length of b", len(b),type(b), b.shape)
# print("length of c", len(c),type(c), c.shape)
# print("length of d", len(d),type(d), d.shape)
# print("length of e", len(e),type(e), e.shape)
# print("length of f", len(f),type(f), f.shape)

# print('done')
# print(get_classnames())