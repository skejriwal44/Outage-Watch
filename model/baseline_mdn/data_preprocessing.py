"""Data Preprocessing Module

This module allows user to preprocess data and splitn into train and test set.

This module contains the following functions:

    * preprocessing: pre-processes data and defines classifier output.
    * train_test: splits data into train and test set for given timesteps.
"""
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def preprocessing(filename_input):
    """Pre-processes data and defines classifier and mdn output layers
    
    Args:.
        filename_input: A string representing the filename of the dataset.
    Returns:
        input_and_labels: A dataframe containing input data and labels.
        classifier_layers: A list containing columns for classifier.
        mdn_layers: A list containing columns for MDN.
    """
    input_and_labels=pd.read_csv(filename_input)
    input_and_labels.dropna(inplace=True)
    input_and_labels.sort_values(by = ['timestamp'],inplace=True)
    input_and_labels.drop(columns=['timestamp'],inplace=True)
    if 'index' in input_and_labels.columns:
        input_and_labels.drop(columns=['index'],inplace=True)
    mdn_layers=[]
    for col in list(input_and_labels.columns):
        if col.endswith('_pred'):
            input_and_labels.drop(columns=col,inplace=True)
    for col in list(input_and_labels.columns):
        if col.endswith('_forecast'):
            mdn_layers.append(col)
    return input_and_labels,mdn_layers

def train_test(input_and_labels,mdn_layers,split,timesteps):
    """Splits data into train and test set for given timesteps.
    
    Args:.
        input_and_labels: A dataframe containing input data and labels.
        mdn_layers: A list containing columns for MDN.
        split: A float representing the train-test split.
        timesteps: An integer representing the timesteps to be taken for input.
    Returns:
        trainX: A numpy array containing input of train data.
        testX: A numpy array containing input of test data.
        y_train_mdn: A list containing mdn output of train data.
        y_test_mdn: A list containing mdn output of test data.
        scalerY: A MinMax scaler object for mdn output data.
    """
    train, test = train_test_split(input_and_labels, test_size=split, random_state=42, shuffle=False)
    x_train = train.drop(columns=mdn_layers,axis=1)
    x_test = test.drop(columns=mdn_layers,axis=1)
    scalerX = MinMaxScaler()
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)

    y_train2 = train.loc[:,mdn_layers]
    y_test2 = test.loc[:, mdn_layers]
    y_train2 = np.array(y_train2)
    y_test2 = np.array(y_test2)
    scalerY = MinMaxScaler()
    y_train2 = scalerY.fit_transform(y_train2)
    train.loc[:,mdn_layers]=y_train2
    y_test2 = scalerY.transform(y_test2)
    test.loc[:,mdn_layers]=y_test2
    trainX = []
    y_train_mdn= []
    for i in range(0, len(x_train)-timesteps+1):
        trainX.append(x_train[i:i+timesteps])
    for metric_i in  mdn_layers:
        y_train_mdn.append(train.loc[timesteps-1:,metric_i])
    trainX = np.array(trainX)
    testX = []
    y_test_mdn = []
    test.reset_index(drop=True,inplace=True)
    for i in range(0, len(x_test)+1-timesteps):
        testX.append(x_test[i:i+timesteps])
    for metric_i in  mdn_layers:
        y_test_mdn.append(test.loc[timesteps-1:,metric_i])
    testX = np.array(testX)
    return trainX,testX,y_train_mdn,y_test_mdn,scalerY