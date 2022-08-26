#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 22:32:16 2022

@author: simonerittenhouse
"""

# Introduction to Machine Learning
# Homework Assignment 4
# 4/23/2022

# importing packages
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn import model_selection, metrics
import torch
import random
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display

# loading data
df = pd.read_csv('/Users/simonerittenhouse/Desktop/Intro to ML/diabetes.csv')

# inspecting data
print(df.columns)
print(df.shape)

# exploratory analysis
correlations = df.corr()
print(correlations)

# setting seed
seed = 4232022
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% PRE-PROCESSING

# recoding gender
df['BiologicalSex'] = np.where(df['BiologicalSex'] == 2, 0, 1)

# Z-scoring non-binary variables
zScore = ['BMI', 'GeneralHealth', 'MentalHealth', 'PhysicalHealth']
for feature in zScore:
    df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

# encoding categorical variables (brackets and zodiac)
categorical = {'AgeBracket': ['18-24','25-29','30-34','35-39','40-44','45-49','50-54',
                              '55-59','60-64','65-69','70-74','75-79','80+'], 
               'EducationBracket': ['only kindergarten','elementary school','some high school',
                                    'GED','some college','college graduate'], 
               'IncomeBracket': ['<$10k', 2, 3, 4, 5, 6, 7, '>$75k'], 
               'Zodiac': ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio',
                          'Sagittarius','Capricorn','Aquarius','Pisces']}
for feature,featureName in categorical.items():
    encoded = pd.get_dummies(df[feature])
    encoded.columns = featureName
    df = pd.concat([df, encoded], axis = 1)
    df = df.drop([feature], axis = 1)
    
print(df.shape)

#%% TRAIN-TEST SPLIT FOR QUESTIONS 1-3

dataDiabetes = df.to_numpy()
train_dataD, test_dataD = model_selection.train_test_split(dataDiabetes, test_size = 0.3, random_state=36)

# diabetes
y_trainD = train_dataD[:,0:1]
y_testD  = test_dataD[:,0:1]

# other features
X_trainD = np.delete(train_dataD, 0, axis = 1)
X_testD = np.delete(test_dataD, 0, axis = 1)

#%% QUESTION ONE

# perceptron using sklearn
percept = Perceptron(tol=1e-3, random_state=0, class_weight='balanced')
percept.fit(X_trainD, y_trainD.ravel())

fpr, tpr, thresholds = metrics.roc_curve(y_testD, percept.decision_function(X_testD))
aucPercept = metrics.auc(fpr, tpr)
print('\nAUC of Perceptron:', aucPercept)
confusion = metrics.confusion_matrix(y_testD, percept.predict(X_testD))
print(confusion)

# plotting ROC curve
plt.figure(figsize = (7,5))
plt.title('Receiver Operating Characteristic: Perceptron')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % aucPercept)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%% QUESTION TWO

# making data tensors
X_train = torch.FloatTensor(X_trainD).to(device)
X_test = torch.FloatTensor(X_testD).to(device)
y_train = torch.FloatTensor(y_trainD).to(device)

# hyperparameters
learning_rate = 1e-2
lambda_l2 = 1e-3
D = X_train.shape[1]
H = 30
C = 1

pos_weight = len(df['Diabetes']) / (2 * len(df[df['Diabetes']==1]))
print('\nPositive Class Weight:', pos_weight)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]))

# storing performance metrics
AUCs_reLU = []
AUCs_sig = []

# training/testing/plotting functions
def FeedForwardTrain(model, criterion, optimizer, y_train=y_train, X_train=X_train):
    model.train()
    for t in range(1000):
        
        # Forward pass over the model to get the logits 
        y_pred = model(X_train)
    
        # Compute the loss
        loss = criterion(y_pred, y_train)
        #auc = metrics.roc_auc_score(y_train, torch.sigmoid(y_pred).detach().numpy())
        #print("[EPOCH]: %i, [LOSS]: %.6f, [AUC]: %.3f" % (t, loss.item(), auc))
        #display.clear_output(wait=True)
    
        # reset (zero) the gradients before running the backward pass over the model
        optimizer.zero_grad()
    
        # Backward pass to compute the gradient of loss w.r.t our learnable params (weights and biases)
        loss.backward()
    
        # Update params
        optimizer.step()
        
def FeedForwardTest(model, y_test=y_testD, X_test=X_test):
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(X_test)).detach().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        auc = metrics.auc(fpr, tpr)
        
        print('\nAUC of', model, '=', auc)
        print(np.unique(np.round(pred), return_counts=True))
        
        return auc, fpr, tpr
    
def ROC_plot(modelName, auc, fpr, tpr):
    plt.figure(figsize = (7,5))
    plt.title('Receiver Operating Characteristic: ' + modelName)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# no activation function - 1 hidden layer
model0 = nn.Sequential(
    nn.Linear(D,H),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model0.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model0, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model0)
ROC_plot('1 Hidden Layer, no activation function', auc, fpr, tpr)
AUCs_reLU.append(auc)
AUCs_sig.append(auc)

# ReLU activation function - 1 hidden layer
model1_relu = nn.Sequential(
    nn.Linear(D,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model1_relu.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model1_relu, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model1_relu)
ROC_plot('1 Hidden Layer, ReLU function', auc, fpr, tpr)
AUCs_reLU.append(auc)

# ReLU activation function - 2 hidden layers
model2_relu = nn.Sequential(
    nn.Linear(D,H),
    nn.ReLU(),
    nn.Linear(H,H),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model2_relu.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model2_relu, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model2_relu)
ROC_plot('2 Hidden Layers, ReLU function', auc, fpr, tpr)
AUCs_reLU.append(auc)

# ReLU activation function - 3 hidden layers
model3_relu = nn.Sequential(
    nn.Linear(D,H),
    nn.ReLU(),
    nn.Linear(H,H),
    nn.Linear(H,H),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model3_relu.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model3_relu, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model3_relu)
ROC_plot('3 Hidden Layers, ReLU function', auc, fpr, tpr)
AUCs_reLU.append(auc)

# Sigmoid activation function - 1 hidden layer
model1_sig = nn.Sequential(
    nn.Linear(D,H),
    nn.Sigmoid(),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model1_sig.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model1_sig, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model1_sig)
ROC_plot('1 Hidden Layer, Sigmoid function', auc, fpr, tpr)
AUCs_sig.append(auc)

# Sigmoid activation function - 2 hidden layers
model2_sig = nn.Sequential(
    nn.Linear(D,H),
    nn.Sigmoid(),
    nn.Linear(H,H),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model2_sig.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model2_sig, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model2_sig)
ROC_plot('2 Hidden Layers, Sigmoid function', auc, fpr, tpr)
AUCs_sig.append(auc)

# Sigmoid activation function - 3 hidden layers
model3_sig = nn.Sequential(
    nn.Linear(D,H),
    nn.Sigmoid(),
    nn.Linear(H,H),
    nn.Linear(H,H),
    nn.Linear(H,C)
    )
optimizer = torch.optim.SGD(model3_sig.parameters(), lr=learning_rate, weight_decay=lambda_l2)

FeedForwardTrain(model3_sig, criterion, optimizer)
auc, fpr, tpr = FeedForwardTest(model3_sig)
ROC_plot('3 Hidden Layers, Sigmoid function', auc, fpr, tpr)
AUCs_sig.append(auc)

#%%

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(len(AUCs_reLU)), AUCs_reLU)
plt.xticks(ticks = np.arange(len(AUCs_reLU)), labels = ['1 layer-no activation', '1 hidden layer', '2 hidden layers', '3 hidden layers'], rotation = 30)
plt.xlabel('Model')
plt.ylabel('AUC')
plt.title('ReLU Models AUC')

plt.subplot(1,2,2)
plt.plot(np.arange(len(AUCs_sig)), AUCs_sig)
plt.xticks(ticks = np.arange(len(AUCs_sig)), labels = ['1 layer-no activation', '1 hidden layer', '2 hidden layers', '3 hidden layers'], rotation = 30)
plt.xlabel('Model')
plt.ylabel('AUC')
plt.title('Sigmoid Models AUC')

plt.show()

#%% QUESTION THREE

train_loaderD = torch.utils.data.DataLoader(torch.FloatTensor(train_dataD), batch_size = 64, shuffle=True)

def train(epoch, model, criterion, train_loader): # takes model, for every epoch we iterate over train loader
    model.train()
    for batch_idx, dataset in enumerate(train_loader):
        # send data to device, where the "device" is either a GPU if it exists or a CPU
        data = dataset[:,1:].to(device)
        target = dataset[:,0:1].to(device)

        optimizer.zero_grad()
        # forward pass through the model
        output = model(data)
        
        if criterion == 'BCELossWithLogits':
            loss = F.binary_cross_entropy_with_logits(input=output, 
                                                      target=target, 
                                                      pos_weight=torch.FloatTensor([pos_weight]))
        elif criterion == 'RMSE':
            loss = torch.sqrt(F.mse_loss(input=output, target=target))
        # backward pass through the cross-entropy loss function and the model
        loss.backward()
        
        optimizer.step()
        #if batch_idx % 100 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(train_loader.dataset),
                #100. * batch_idx / len(train_loader), loss.item()))

# deep feedforward network
class FC4Layer(nn.Module): # nn.Module is a parent class
    def __init__(self, input_size, n_hidden, output_size):
        super(FC4Layer, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden), 
            nn.ReLU(), 
            nn.Linear(n_hidden, n_hidden), 
            nn.ReLU(), 
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size))

    def forward(self, x): # define forward call (x is data)
        x = x.view(-1, self.input_size)        
        return self.network(x)

# deep feed-forward network
model_fnn = FC4Layer(X_train.shape[1], 30, 1)
model_fnn.to(device)
optimizer = optim.SGD(model_fnn.parameters(), lr=1e-2, weight_decay=1e-3)

for epoch in range(100):
    train(epoch, model_fnn, 'BCELossWithLogits', train_loaderD)
    
#%%

auc, fpr, tpr = FeedForwardTest(model_fnn)
ROC_plot('4 Hidden Layers, ReLU function', auc, fpr, tpr)
# confusion matrix
with torch.no_grad():
    preds = torch.round(torch.sigmoid(model_fnn(X_test))).detach().numpy()
    confusionFNN = metrics.confusion_matrix(y_testD, preds)
    print(confusionFNN)
        
#%% QUESTION FOUR

# getting raw BMI
orig = pd.read_csv('/Users/simonerittenhouse/Desktop/Intro to ML/diabetes.csv')
df['BMI'] = orig['BMI']

dataBMI = df.to_numpy()

# train-test split
train_dataB, test_dataB = model_selection.train_test_split(dataBMI, test_size = 0.3, random_state=36)

# BMI
y_trainB = train_dataB[:,3:4]
y_testB  = test_dataB[:,3:4]

# other features
X_trainB = np.delete(train_dataB, 3, axis = 1)
X_testB = np.delete(test_dataB, 3, axis = 1)

# making data tensors
X_train = torch.FloatTensor(X_trainB).to(device)
X_test = torch.FloatTensor(X_testB).to(device)
y_train = torch.FloatTensor(y_trainB).to(device)

# hyperparameters
learning_rate = 1e-2
lambda_l2 = 1e-5
D = X_train.shape[1]
H = 30
C = 1 # single prediction of BMI

criterion = nn.MSELoss() # minimize RMSE (need to square root this for evaluation)

# storing performance metrics
RMSE = []

# training/testing/plotting functions
def RMSETrain(model, criterion, optimizer, epochs=1000, y_train=y_train, X_train=X_train, sqrt=True):
    model.train()
    for t in range(epochs):
    
        # Forward pass over the model to get the logits 
        y_pred = model(X_train)
    
        if sqrt==True:
            # Compute the loss (sqrt MSE)
            loss = torch.sqrt(criterion(y_pred, y_train))
        else:
            loss = criterion(y_pred, y_train)
        #print("[EPOCH]: %i, [LOSS]: %.6f" % (t, loss.item()))
        #display.clear_output(wait=True)
    
        # reset (zero) the gradients before running the backward pass over the model
        optimizer.zero_grad()
    
        # Backward pass to compute the gradient of loss w.r.t our learnable params (weights and biases)
        loss.backward()
    
        # Update params
        optimizer.step()

def RMSETest(model, y_test=y_testB, X_test=X_test):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).detach().numpy()
        rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
        
        print('\nRMSE of', model, '=', rmse)
        
        return rmse

# no activation function
model_NA = nn.Sequential(
    nn.Linear(D,H),
    nn.Linear(H,C)
    )
optimizer = torch.optim.Adam(model_NA.parameters(), lr=learning_rate, weight_decay=lambda_l2)

RMSETrain(model_NA, criterion, optimizer)
RMSE.append(RMSETest(model_NA))

# RELU activation function
model_relu = nn.Sequential(
    nn.Linear(D,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )
optimizer = torch.optim.Adam(model_relu.parameters(), lr=learning_rate, weight_decay=lambda_l2)

RMSETrain(model_relu, criterion, optimizer)
RMSE.append(RMSETest(model_relu))

# Tanh activation function
model_tanh = nn.Sequential(
    nn.Linear(D,H),
    nn.Tanh(),
    nn.Linear(H,C)
    )
optimizer = torch.optim.Adam(model_tanh.parameters(), lr=learning_rate, weight_decay=lambda_l2)

RMSETrain(model_tanh, criterion, optimizer)
RMSE.append(RMSETest(model_tanh))

# Sigmoid activation function
model_sig = nn.Sequential(
    nn.Linear(D,H),
    nn.Sigmoid(),
    nn.Linear(H,C)
    )
optimizer = torch.optim.Adam(model_sig.parameters(), lr=learning_rate, weight_decay=lambda_l2)

RMSETrain(model_sig, criterion, optimizer)
RMSE.append(RMSETest(model_sig))

# ELU activation function
model_elu = nn.Sequential(
    nn.Linear(D,H),
    nn.ELU(),
    nn.Linear(H,C)
    )
optimizer = torch.optim.Adam(model_elu.parameters(), lr=learning_rate, weight_decay=lambda_l2)

RMSETrain(model_elu, criterion, optimizer)
RMSE.append(RMSETest(model_elu))

plt.figure(figsize = (7,5))
plt.plot(np.arange(len(RMSE)),RMSE)
plt.plot(RMSE.index(min(RMSE)), RMSE[RMSE.index(min(RMSE))], marker="o", markersize=10)
plt.xlabel('Activation Function Used')
plt.ylabel('RMSE')
plt.title('RMSE Across Models for BMI Prediction')
plt.xticks(ticks = np.arange(len(RMSE)), 
           labels=['None', 'ReLU', 'Tanh', 'Sigmoid', 'ELU'], 
           rotation = 50)
plt.show()

#%% QUESTION FIVE

# baseline hyperparameters
learning_rate = 1e-2
lambda_l2 = 1e-5
D = X_train.shape[1]
H = 30
C = 1 # single prediction of BMI

RMSE_STRUCT = []
models_STRUCT = []
modelName_STRUCT = []

# testing structure (number of hidden layers, number of H neurons, activation function)
model_0 = nn.Sequential(
    nn.Linear(D,H),
    nn.Tanh(),
    nn.Linear(H,H),
    nn.Tanh(),
    nn.Linear(H,H),
    nn.Tanh(),
    nn.Linear(H,C)
    )
models_STRUCT.append(model_0)
modelName_STRUCT.append('Tanh-3 H layers')

model_1 = nn.Sequential(
    nn.Linear(D,H),
    nn.ReLU(),
    nn.Linear(H,H),
    nn.ReLU(),
    nn.Linear(H,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )
models_STRUCT.append(model_1)
modelName_STRUCT.append('ReLU-3 H layers')

model_2 = nn.Sequential(
    nn.Linear(D,H*2),
    nn.ReLU(),
    nn.Linear(H*2,H*3),
    nn.ReLU(),
    nn.Linear(H*3,H*2),
    nn.ReLU(),
    nn.Linear(H*2,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )
models_STRUCT.append(model_2)
modelName_STRUCT.append('ReLU-4 H layers, variable H neurons')

model_3 = nn.Sequential(
    nn.Linear(D,H*2),
    nn.Tanh(),
    nn.Linear(H*2,H*3),
    nn.Tanh(),
    nn.Linear(H*3,H*2),
    nn.Tanh(),
    nn.Linear(H*2,H),
    nn.Tanh(),
    nn.Linear(H,C)
    )
models_STRUCT.append(model_3)
modelName_STRUCT.append('Tanh-4 H layers, variable H neurons')

model_4 = nn.Sequential(
    nn.Linear(D,H*2),
    nn.ReLU(),
    nn.Linear(H*2,H*3),
    nn.ReLU(),
    nn.Linear(H*3,H*4),
    nn.ReLU(),
    nn.Linear(H*4,H*3),
    nn.ReLU(),
    nn.Linear(H*3,H*2),
    nn.ReLU(),
    nn.Linear(H*2,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )
models_STRUCT.append(model_4)
modelName_STRUCT.append('ReLU-6 H layers, variable H neurons')

criterion = nn.MSELoss()
for model in models_STRUCT:
    RMSETrain(model, 
              criterion, 
              optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2), 
              epochs=100)
    RMSE_STRUCT.append(RMSETest(model))

# plotting change in RMSE
RMSEStructPlt = pd.DataFrame({'RMSE':RMSE_STRUCT,
                            'Model':modelName_STRUCT})
RMSEStructPlt.sort_values(by=['RMSE'], ascending=False, inplace=True)
    
plt.figure(figsize=(10,5))
plt.bar(range(len(models_STRUCT)), RMSEStructPlt['RMSE'])
plt.ylim([5.8, 6.8])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(ticks = range(len(models_STRUCT)),
           labels = RMSEStructPlt['Model'],
           rotation = 50)
plt.title('RMSE While Varying Model Structure')
plt.show()

#%%

# testing learning rate, weight_decay
RMSE_PARAM = []
modelName_PARAM = []

for lr in [1e-2, 1e-3, 1e-4]:
    for wt_decay in [0, 1e-5, 1e-3, 1e-2]:
        model_ = nn.Sequential(
            nn.Linear(D,H*2),
            nn.ReLU(),
            nn.Linear(H*2,H*3),
            nn.ReLU(),
            nn.Linear(H*3,H*4),
            nn.ReLU(),
            nn.Linear(H*4,H*3),
            nn.ReLU(),
            nn.Linear(H*3,H*2),
            nn.ReLU(),
            nn.Linear(H*2,H),
            nn.ReLU(),
            nn.Linear(H,C)
            )
        RMSETrain(model_,
                  criterion,
                  optimizer = torch.optim.Adam(model_.parameters(), lr=lr, weight_decay=wt_decay),
                  epochs = 100)
        RMSE_PARAM.append(RMSETest(model_))
        modelName_PARAM.append('lr = {}, wt_decay = {}'.format(lr, wt_decay))

# plotting change in RMSE
RMSEParamPlt = pd.DataFrame({'RMSE':RMSE_PARAM,
                            'Model':modelName_PARAM})
RMSEParamPlt.sort_values(by=['RMSE'], ascending=False, inplace=True)
    
plt.figure(figsize=(10,5))
plt.bar(range(len(modelName_PARAM)), RMSEParamPlt['RMSE'])
plt.ylim([5.8, 30])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(ticks = range(len(modelName_PARAM)),
           labels = RMSEParamPlt['Model'],
           rotation = 90)
plt.title('RMSE While Varying Hyperparameters')
plt.show()

#%%

# testing criterion, optimizer
RMSE_OPT = []
modelName_OPT = []

lr = 1e-2
lambda_l2 = 1e-3

criterion = nn.MSELoss()
for opt in range(2):
    model_ = nn.Sequential(
        nn.Linear(D,H*2),
        nn.ReLU(),
        nn.Linear(H*2,H*3),
        nn.ReLU(),
        nn.Linear(H*3,H*4),
        nn.ReLU(),
        nn.Linear(H*4,H*3),
        nn.ReLU(),
        nn.Linear(H*3,H*2),
        nn.ReLU(),
        nn.Linear(H*2,H),
        nn.ReLU(),
        nn.Linear(H,C)
        )
    if opt == 0:
        optimizer = torch.optim.Adam(model_.parameters(), lr=lr, weight_decay=lambda_l2)
        name = 'Adam'
    else:
        optimizer = torch.optim.SGD(model_.parameters(), lr=lr, weight_decay=lambda_l2)
        name = 'SGD'

    for crit in range(2):
        if crit == 0:
            RMSETrain(model_,
                      criterion,
                      optimizer = optimizer,
                      epochs=100,
                      sqrt = False)
            RMSE_OPT.append(RMSETest(model_))
            modelName_OPT.append("Opt= {}, Crit= MSELoss".format(name))
        else:
            RMSETrain(model_,
                      criterion,
                      optimizer = optimizer,
                      epochs=100,
                      sqrt = True)
            RMSE_OPT.append(RMSETest(model_))
            modelName_OPT.append("Opt= {}, Crit= RMSELoss".format(name))
            
# plotting change in RMSE
RMSEOptPlt = pd.DataFrame({'RMSE':RMSE_OPT,
                            'Model':modelName_OPT})
RMSEOptPlt.sort_values(by=['RMSE'], ascending=False, inplace=True)
    
plt.figure(figsize=(10,5))
plt.bar(range(len(modelName_OPT)), RMSEOptPlt['RMSE'])
plt.ylim([5.8, 8])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(ticks = range(len(modelName_OPT)),
           labels = RMSEOptPlt['Model'],
           rotation = 10)
plt.title('RMSE While Varying Optimizer and Loss Criterion')
plt.show()

#%%

# Increasing epochs for final model
finalBMI = nn.Sequential(
    nn.Linear(D,H*2),
    nn.ReLU(),
    nn.Linear(H*2,H*3),
    nn.ReLU(),
    nn.Linear(H*3,H*4),
    nn.ReLU(),
    nn.Linear(H*4,H*3),
    nn.ReLU(),
    nn.Linear(H*3,H*2),
    nn.ReLU(),
    nn.Linear(H*2,H),
    nn.ReLU(),
    nn.Linear(H,C)
    )

lr = 1e-2
lambda_l2 = 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(finalBMI.parameters(), lr=lr, weight_decay=lambda_l2)

# 1000 epochs, criterion = RMSELoss
RMSETrain(finalBMI, criterion, optimizer)
RMSETest(finalBMI)

#%% EXTRA CREDIT A

# accuracy for diabetes models 
def get_feature_importance(model, feature, n, X_test=torch.FloatTensor(X_testD).to(device), y_test=y_testD):
    model.eval()
    global pred
    with torch.no_grad():
        pred = torch.round(torch.sigmoid(model(X_test))).detach().numpy()
        s = metrics.accuracy_score(y_test, pred) # baseline score
    total = 0.0
    for i in range(n):
        perm = np.random.permutation(range(X_test.shape[0]))
        X_test_ = X_test.numpy().copy()
        X_test_[:, feature] = X_test[perm, feature]
        X_test_ = torch.FloatTensor(X_test_).to(device)
        model.eval()
        with torch.no_grad():
                pred_ = torch.round(torch.sigmoid(model(X_test_))).detach().numpy()
                s_i = metrics.accuracy_score(y_test, pred_)
        total += s_i
    return s - total / n

fImpDia1 = []
for feature in range(X_testD.shape[1]):
    fImpDia1.append(get_feature_importance(model0, 
                   feature, 50))
    
fImpDia2 = []
for feature in range(X_testD.shape[1]):
    fImpDia2.append(get_feature_importance(model3_relu, 
                   feature, 50))
    
#%%

# finding least important features
diabetesImp1 = pd.DataFrame({'Diabetes Classification':fImpDia1,
                            'Diabetes Feature':list(df.drop(['Diabetes'], axis=1).columns)})
diabetesImp1.sort_values(by=['Diabetes Classification'], inplace=True)

diabetesImp2 = pd.DataFrame({'Diabetes Classification':fImpDia2,
                            'Diabetes Feature':list(df.drop(['Diabetes'], axis=1).columns)})
diabetesImp2.sort_values(by=['Diabetes Classification'], inplace=True)

print('\nLeast Important Features: Diabetes Classification (No Activation)')
print(diabetesImp1.head(20))

print('\nLeast Important Features: Diabetes Classification (ReLU)')
print(diabetesImp2.head(20))

# plotting
plt.figure(figsize=(20, 25))
plt.subplot(2,1,1)
plt.bar(range(len(diabetesImp1)), diabetesImp1['Diabetes Classification'], color="r", alpha=0.7)
plt.xticks(ticks=range(len(diabetesImp1)), 
           labels = diabetesImp1['Diabetes Feature'], rotation = 70)
plt.ylabel("Importance")
plt.title("Feature importances: Diabetes Classification (No Activation)")

plt.subplot(2,1,2)
plt.bar(range(len(diabetesImp2)), diabetesImp2['Diabetes Classification'], color="r", alpha=0.7)
plt.xticks(ticks=range(len(diabetesImp2)), 
           labels = diabetesImp2['Diabetes Feature'], rotation = 70)
plt.ylabel("Importance")
plt.title("Feature importances: Diabetes Classification (ReLU)")
plt.show()

#%% EXTRA CREDIT B

# please see written report for statement
