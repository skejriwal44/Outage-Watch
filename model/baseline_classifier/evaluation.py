"""Evaluation Module

This module allows the user to evaluate the classifier model on basis of 
classification AUC.

This module contains the following functions:

    * generate_roc: returns fpr,tpr and thresholds for given y_test and y_pred.
    * generate_youden: calculates optimum thresholds using youden statistics.
    * validate_model: calculates classification auc of model and stores results.
"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_curve,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import scipy.stats as st
from operations import slice_parameter_vectors
from sklearn.metrics import auc


def generate_roc(y_test,y_pred):
    """Returns FPR,TPR and thresholds for given pair of y_test and y_pred.
    
    Args:.
        y_test: A list containing real labels.
        y_pred: A list containing predicted labels.
    Returns:
        fpr: An integer representing False Positivite Rate.
        tpr: An integer representing True Positive Rate.
        thresholds: An integer representing thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    return tpr,fpr,thresholds


def generate_youden(tpr,fpr,thresholds):
    """Calculates optimum thresholds using Youden Statistics.
    
    Args:.
        fpr: An integer representing False Positivite Rate.
        tpr: An integer representing True Positive Rate.
        thresholds: An integer representing thresholds.
    Returns:
        thresholdOpt: An integer representing optimal threshold.
    """
    youdenJ = tpr - fpr
    index = np.argmax(youdenJ)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    youdenJOpt = round(youdenJ[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    return thresholdOpt

def validate_model(model,history,testX,classifier_layers,y_test,file_det,y_train,no_of_metrics,tgt_dir):
    """Calculates and stores results of Classifier.
    
    Args:
        model: A Tensorflow Model build using the proposed architecture.
        history: An object representing the history of model run.
        testX: A numpy array containing input of test data.
        classifier_layers: A list containing column name of classifier layers.
        y_test: A list containing classifier output of test data.
        file_det: A string representing the file name to store results.
        y_train: A list containing classifier output of train data.
        no_of_metrics: An integer representing number of predicted labels.
        tgt_dir: A string representing target directory to store csv.
    Returns:
        None
    """
    prediction_output=model.predict(testX)
    y_pred = prediction_output[0:no_of_metrics]
    for i in range(len(y_test)):
        y_test[i] = y_test[i].to_numpy()
    cutoff = {'youdenJ': []}
    for i in range(len(classifier_layers)):
        tpr,fpr,thresholds = generate_roc(y_test[i],y_pred[i])
        cutoff['youdenJ'].append(generate_youden(tpr,fpr,thresholds))
    data=[]
    for i in range(len(classifier_layers)):
        data.append([])
        train_name = 'output_'+ classifier_layers[i]+'_auc'
        test_name = 'val_output_'+ classifier_layers[i]+'_auc'
        y_pred_=prediction_output[i][:].flatten()>=cutoff['youdenJ'][i]
        y_test_=y_test[i]==1
        cm=confusion_matrix(y_test_,y_pred_)
        if(len(cm)==2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn=cm.ravel()[0]
            fp=0
            fn=0
            tp=0
        data[i].append(classifier_layers[i])
        data[i].append(history.history[train_name][-1])
        data[i].append(history.history[test_name][-1])
        data[i].append(str(round(sum(y_train[i])/len(y_train[i])*100,2))+"%")
        data[i].append(str(round(sum(y_test[i])/len(y_test[i])*100,2)) +"%")
        data[i].append(tp)
        data[i].append(fp)
        data[i].append(tn)
        data[i].append(fn)
    df_analysis = pd.DataFrame(data,columns=['Golden Metric','Training AUC', 'Test AUC', 'Training Class Imbalance', 'Test Class Imbalance','True Positive','False Positive','True Negative','False Negative'])
    df_analysis.to_csv(tgt_dir+file_det+'classifier_auc.csv',index=False)

