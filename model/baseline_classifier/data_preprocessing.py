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
    """Pre-processes data and defines classifier output.
    
    Args:.
        filename_input: A string representing the filename of the dataset.
    Returns:
        input_and_labels: A dataframe containing input data and labels.
        classifier_layers: A list containing columns for classifier.
    """
    input_and_labels=pd.read_csv(filename_input)
    input_and_labels.dropna(inplace=True)
    input_and_labels.sort_values(by = ['timestamp'],inplace=True)
    input_and_labels.drop(columns=['timestamp'],inplace=True)
    if 'index' in input_and_labels.columns:
        input_and_labels.drop(columns=['index'],inplace=True)
    classifier_layers=[]
    for col in list(input_and_labels.columns):
        if col.endswith('_pred'):
            classifier_layers.append(col)
    for col in list(input_and_labels.columns):
        if col.endswith('_forecast'):
            input_and_labels.drop(columns=col,inplace=True)
    return input_and_labels,classifier_layers

def train_test(input_and_labels,classifier_layers,timesteps,split):
    """Splits data into train and test set for given timesteps.
    
    Args:.
        input_and_labels: A dataframe containing input data and labels.
        classifier_layers: A list containing columns for classifier.
        timesteps: An integer representing the timesteps to be taken for input.
    Returns:
        trainX: A numpy array containing input of train data.
        testX: A numpy array containing input of test data.
        y_train_classifier: A list containing classifier output of train data.
        y_test_classifier: A list containing classifier output of test data.
    """
    train, test = train_test_split(input_and_labels, test_size=split, random_state=42, shuffle=False)
    x_train = train.drop(columns=classifier_layers,axis=1)
    x_test = test.drop(columns=classifier_layers,axis=1)
    scalerX = MinMaxScaler()
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)

    trainX = []
    y_train_classifier = []
    for i in range(0, len(x_train)-timesteps+1):
        trainX.append(x_train[i:i+timesteps])
    for metric_i in  classifier_layers:
        y_train_classifier.append(train.loc[timesteps-1:,metric_i])
    trainX = np.array(trainX)
    testX = []
    y_test_classifier = []
    test.reset_index(drop=True,inplace=True)
    for i in range(0, len(x_test)+1-timesteps):
        testX.append(x_test[i:i+timesteps])
    for metric_i in  classifier_layers:
        y_test_classifier.append(test.loc[timesteps-1:,metric_i])
    testX = np.array(testX)
    return trainX,testX,y_train_classifier,y_test_classifier