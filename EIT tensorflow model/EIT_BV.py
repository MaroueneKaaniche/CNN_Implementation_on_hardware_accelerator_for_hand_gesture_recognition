'''
By Malak Fora 18/3/2021
EIT boundary voltage
'''

# *********************************************************************************************************************
# Imports
# *********************************************************************************************************************
import time
import datetime
import os
import random
import pandas as pd
import numpy as np
from numpy import linalg as LA
np.random.seed(1337) # for reproducibility

from keras.layers import Input,Dense, Conv1D, MaxPool1D, Flatten, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
callbacks=[EarlyStopping(patience=10, mode='min', monitor= 'loss') ]#, restore_best_weights = 'TRUE')
from keras.utils.vis_utils import plot_model #to save image model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score #for manual measurement of accuracy using predicted labels


import pickle
import matplotlib.pyplot as plt

import itertools
from sklearn.metrics import confusion_matrix
# *********************************************************************************************************************
# Function to Find the number of train and testing labels for each class in each set
# *********************************************************************************************************************
def counting(dataset):
    frequencies = {}
    for item in dataset:
        if item in frequencies:
            frequencies[item] += 1
        else:
            frequencies[item] = 1
    return frequencies


# *********************************************************************************************************************
# To know the coloumn location in the matrix for the first sensor for each trial
# *********************************************************************************************************************
tr = []
count = 1
for s in range(0, 10):  # i=1:10  #to catch all trials
    tr.append((9 * count))
    count = count + 1

                                    # step1: Form dataset & make the alternate order
# *********************************************************************************************************************
# Alternative order / original FMG before smoothing Or smoothed FMG , according to the selected path
# *********************************************************************************************************************

data = []
data = np.zeros((800 , 40+ 1))

r = 0  # r is row of the data matrix
for g in range(0, 10):   # 0<=i>10 # 10 gestures
    for p in range(1, 9):  #8 persons

        df = pd.read_excel( r'D:\Thesis\EIT BV\EIT Boundary Voltage\p'+str(p)+'\BV'+str(p)+'_'+str(g)+'.xlsx', header=None)
        df = df.values

        for i in range(0, 10):  #trial 1-10
            data[r, 0] = g;
            data[r, 1:]=df[:,i+1]/LA.norm(df[:,i+1])  #first coloumn in the BV excel sheet is refernce
            r = r + 1;


print('data shape: ', data.shape)

#                                  Step 2: Data splitting (80% train/val and 20% testing)
# *********************************************************************************************************************
# splitting into train and testing 1st 8 for training, last 2 trials testing
# *********************************************************************************************************************
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

print('length of training data: ', len(X))
print('length of the training labels: ', len(Ytr))
print('length of the testing data: ', len(Xtest))
print('length of the testing labels: ', len(Yts))

#note: data shuffling will be after dividing X into train and validation using stratified K fold Cross validation

#                                  Step 3: Data reshaping to be suitable for 1D CNN model
# *********************************************************************************************************************
# reshape training and testing FMG signals to be suitable for 1D CNN
# *********************************************************************************************************************

X_fmg_train = np.array(X)
X_fmg_train = X_fmg_train.reshape(len(X_fmg_train), X_fmg_train.shape[1], 1)  # len(X_fmg_train) or X_fmg_train.shape[0]
X_fmg_test = np.array(Xtest)
X_fmg_test = X_fmg_test.reshape(len(X_fmg_test), X_fmg_test.shape[1], 1)


# *********************************************************************************************************************
#  model architecture
# *********************************************************************************************************************

TrainingTime=[]
train_val_loss= []
train_val_accuracy=[]
train_loss_lastEpoch=[] #Training loss value for the last epoch
train_acc_lastEpoch=[] # Training Accuracy value for the last epoch
val_loss_lastEpoch=[] #validation loss value for the last epoch
val_acc_lastEpoch=[] # validation Accuracy value for the last epoch
no_round_epochs=[] # number of epochs requirerd for training the model per fold
test_loss=[] #per fold using .evaluate
test_accuracy=[] #per fold using .evaluate
testAcc_predictions=[] #per fold using predicted labels
predictions=[]

num_of_folds= 5
kfold = StratifiedKFold(n_splits=num_of_folds, shuffle=True,random_state=123)

Y= np.array(Ytr) #convert Y from list to numpy in order to be able to take multiple indixes from it at the same time
fold_num=1

t1_AllFolds = time.time() #starting time for all folds
for train_idx, val_idx in kfold.split(X_fmg_train,Y):

    random.seed(4)
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    tf.keras.backend.clear_session()
    t1 = time.time()  # training time for each fold




    sig_shape = (X_fmg_train.shape[1], 1)
    Signal_Inputs = Input(shape=sig_shape, name='Signal_Inputs')
    conv1 = Conv1D(filters=8, kernel_size= 3, activation='relu', input_shape=sig_shape)(Signal_Inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1)
    D=Dropout(0.2)(pool1)

    flat1 = Flatten()(D)
    hidden1 = Dense(128,activation='relu')(flat1)
    hidden2 = Dense(64,activation='relu')(hidden1)
    output = Dense(10, activation='softmax')(hidden2)


    from keras import metrics
    fmgmodel = Model(inputs=Signal_Inputs, outputs=output)
    opt =  keras.optimizers.Adam (lr = 0.001)
    fmgmodel.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = fmgmodel.fit(X_fmg_train[train_idx], Y[train_idx], epochs=250, batch_size=32,
                           validation_data=(X_fmg_train[val_idx], Y[val_idx]),
                           callbacks=callbacks)  # good batch: 32,128 bad: 256, 512

    t2 = time.time() - t1
    TrainingTime.append(round(t2 / 60, 2))
    # >>>> Learning curves (Accuracy and loss) <<<<
    train_val_loss.append([fold_num, history.history['loss'], history.history['val_loss']])
    train_val_accuracy.append([fold_num, history.history['accuracy'], history.history['val_accuracy']])

    train_loss_lastEpoch.append(round(train_val_loss[fold_num - 1][1][-1], 5))
    val_loss_lastEpoch.append(round(train_val_loss[fold_num - 1][2][-1], 5))
    train_acc_lastEpoch.append(round(100 * train_val_accuracy[fold_num - 1][1][-1], 2))
    val_acc_lastEpoch.append(round(100 * train_val_accuracy[fold_num - 1][2][-1], 2))
    no_round_epochs.append(len(history.history['loss']))

    '''
    # >>> save the model per fold <<<
    now = datetime.datetime.now()
    picklename = "eitBV" + str(fold_num)+ "__" + str(now.hour) + "_" + str( now.minute) + ".pkl"
    path="D:/Thesis/g_hybrid/bv/"
    with open(path+picklename, 'wb') as fid:
        pickle.dump(fmgmodel, fid)
    '''

    testing_score= fmgmodel.evaluate(X_fmg_test, Yts)
    test_loss.append(testing_score[0])
    test_accuracy.append(testing_score[1]*100)

    # >>> Model.predict testing data <<<
    pred=fmgmodel.predict(X_fmg_test)
    pred=pred.argmax(axis=1)
    predictions.append([fold_num,pred])
    testAcc_predictions.append(round(100*accuracy_score(pred, Yts),2))

    print(f'score for fold {fold_num}: {fmgmodel.metrics_names[0]} of {testing_score[0]}: {fmgmodel.metrics_names[1]} of {testing_score[1]*100}%')
    fold_num += 1    #new fold


t2_AllFolds = time.time() - t1_AllFolds
TrainingTime.append(round(t2_AllFolds/60,2))






#=====================
for fold in range (num_of_folds):
    #plt.subplot(3,2,fold+1)
    plt.figure()
    plt.plot(train_val_loss[fold][1])
    plt.plot(train_val_loss[fold][2]) #Val
    plt.legend(['Training', 'Validation'])
    plt.xlabel('epochs')
    plt.ylabel('Loss value')
    plt.title ("Loss fold "+str(fold+1))

    figure = plt.gcf()
    figure.set_size_inches(8,7)
    #plt.savefig(path+"LossCurve_fold"+str(fold+1)+".tif")
    plt.close("all")


#plt.figure()
for fold in range (num_of_folds):
    #plt.subplot(3,2,fold+1)
    plt.figure()
    plt.plot(100*np.array(train_val_accuracy[fold][1]))
    plt.plot(100*np.array(train_val_accuracy[fold][2])) #Val
    plt.legend(['Training', 'Validation'])
    plt.xlabel('epochs')
    plt.ylabel('Accuracy value')
    plt.title ("Accuracy fold "+str(fold+1))

    figure = plt.gcf()
    figure.set_size_inches(8,7)
    #plt.savefig(path+str(fold+1)+".tif")







# *********************************************************************************************************************
#   Results per fold
# *********************************************************************************************************************

print('------------------------------------------------------------------------')
print(' >>>> Scores per fold, using model.evaluate <<<<')
for i in range(0, len(test_accuracy)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {round(test_loss[i],3)} - Accuracy: {round(test_accuracy[i],2)}%')

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {round(np.mean(test_accuracy),2)} (+- {round(np.std(test_accuracy),2)})')
print(f'> Loss: {round(np.mean(test_loss),3)}')
print('------------------------------------------------------------------------')

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('Results per fold')
print('Training time in min:', TrainingTime)
print('Number of epochs per fold:', no_round_epochs)
print('Last epoch training accuracy:', train_acc_lastEpoch)
print('Last epoch validation accuracy:', val_acc_lastEpoch)
print('Last epoch training loss:', train_loss_lastEpoch)
print('Last epoch validation loss:', val_loss_lastEpoch)
print('Accuracy using manual prediction:', testAcc_predictions)
print(f'> Average Accuracy: {round(np.mean(testAcc_predictions),2)} (+- {round(np.std(testAcc_predictions),2)})')

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

'''
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(' >>>> Scores per fold, using model.predict<<<<')
print("classification reports")
for fold in range(0, num_of_folds):
    print('===============================')
    print('classification report, fold'+str(fold+1))
    predic=predictions[fold][1]
    print(classification_report(Yts, predic))
    # >>> Confusion matrix
    Categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    axis_font = {'fontname': 'Arial', 'size': '15'}
    cm = confusion_matrix(Yts, predic, Categories)
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    #print(norm_cm)

    plt.figure()
    plt.imshow(norm_cm, interpolation='nearest', cmap=plt.get_cmap('summer'))
    plt.colorbar()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(norm_cm[i, j]), fontsize=13,horizontalalignment='center')
    plt.xticks(Categories, fontsize=14)
    plt.yticks(Categories, fontsize=14)
    plt.ylim([9.5, -0.5])
    plt.tight_layout()
    plt.xlabel("Predicted class", axis_font)
    plt.ylabel("Actual class", axis_font)
    plt.title("Confusion matrix fold " + str(fold + 1))
   
    figure = plt.gcf()
    figure.set_size_inches(7, 7)
    #plt.savefig(path + "Confmat_fold" + str(fold + 1) + ".tif")
    plt.close("all")

'''
