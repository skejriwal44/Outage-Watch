"""Evaluation Module

This module allows the user to evaluate performance of the model in terms of MDN AUC.

This module contains the following functions:

    * generate_roc: returns fpr,tpr and thresholds for y_test and y_pred.
    * generate_youden: calculates optimum threshold using youden statistics.
    * store_loss: calculates and stores the mdn loss.
    * prob_calc_u: calculates upper threshold crossing probability.
    * prob_calc_l: calculates lower threshold crossing probability.
    * mdn_auc_score: calculates and stores MDN AUC.
    * generate_test_data: generates test data for evaluation.
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
    """Returns FPR,TPR and thresholds for given pait of y_test and y_pred.
    
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

def validate_model(model,history,testX,classifier_layers,y_test,file_det,y_train,pred_size,tgt_dir):
    """Calculates and stores results of Classifier.
    
    Args:
        model: A Tensorflow Model build using the proposed architecture.
        history: An object representing the history of model run.
        testX: A numpy array containing input of test data.
        classifier_layers: A list containing column name of classifier layers.
        y_test: A list containing classifier output of test data.
        file_det: A string representing the file name to store results.
        y_train: A list containing classifier output of train data.
        pred_size: An integer representing number of predicted labels.
        tgt_dir: A string representing target directory to store csv.
    Returns:
        None
    """
    prediction_output=model.predict(testX)
    y_pred = prediction_output[0:pred_size]
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

    
def store_loss(history,file_det,tgt_dir):
    """Calculates and stores results of MDN loss.
    
    Args:
        history: An object representing the history of model run.
        file_det: A string representing the file name to store results.
        tgt_dir: A string representing target directory to store csv.
    Returns:
        None
    """
    cols = []
    val_loss = []
    for i in history.history:
        print(i)
        if i.startswith('val') and i.endswith('forecast_loss'):
            cols.append(('_').join(i.split('_')[1:]))
            val_loss.append(history.history[i][-1])
    df = pd.DataFrame(columns = cols)
    df.loc[len(df)] = val_loss
    print(df)
    df.to_csv(tgt_dir+file_det+'mdn_loss.csv',index=False)
    
def prob_calc_u(alpha_pred,mu_pred,sigma_pred,thresh):
    """Calculates P(x>tresh_u) for gaussian mixture model(right tail).
    
    Args:
        alpha_pred: A list containing the mixture coefficients of distributions.
        mu_pred: A list containing mean of distributions.
        sigma_pred: A list containing std dev of distribution.
        thresh: An integer representing the upper threshold.
    Returns:
        probab_list: A list cotaining probability of value crossing upper threshold.
    """
    probab_list=[]
    for i in range(len(alpha_pred)):
        x1=1-norm.cdf(x=thresh,loc=mu_pred[i][0],scale=sigma_pred[i][0])
        x2=1-norm.cdf(x=thresh,loc=mu_pred[i][1],scale=sigma_pred[i][1])
        x3=1-norm.cdf(x=thresh,loc=mu_pred[i][2],scale=sigma_pred[i][2])
        probab_list.append(alpha_pred[i][0]*x1+alpha_pred[i][1]*x2+alpha_pred[i][2]*x3)
    return probab_list

def prob_calc_l(alpha_pred,mu_pred,sigma_pred,thresh):
    """Calculates P(x<tresh_l) for gaussian mixture model(left tail).
    
    Args:
        alpha_pred: A list containing the mixture coefficients of distributions.
        mu_pred: A list containing mean of distributions.
        sigma_pred: A list containing std dev of distribution.
        thresh: An integer representing the lower threshold.
    Returns:
        probab_list: A list cotaining probability of value remaining below lower threshold.
    """
    probab_list=[]
    for i in range(len(alpha_pred)):
        x1=norm.cdf(x=thresh,loc=mu_pred[i][0],scale=sigma_pred[i][0])
        x2=norm.cdf(x=thresh,loc=mu_pred[i][1],scale=sigma_pred[i][1])
        x3=norm.cdf(x=thresh,loc=mu_pred[i][2],scale=sigma_pred[i][2])
        probab_list.append(alpha_pred[i][0]*x1+alpha_pred[i][1]*x2+alpha_pred[i][2]*x3)
    return probab_list

def mdn_auc_score(input_and_labels,tgt_dir,file_det,scalerY,mdn_layers,model,testX,y_test,thresh_percentage,pred_window,tails,no_of_metrics,components):
    """Calculates and stores the MDN AUC.
    
    Args:
        input_and_labels:A dataframe containing input data and labels.
        tgt_dir: A string containing the directory where the results are to be stored.
        file_det: A string containing the destination where file is to be stored.
        scalerY: A MinMax scaler object for mdn output data.
        mdn_layers: A list containing columns for MDN.
        model: A Tensorflow Model build using the proposed architecture.
        testX: A dataframe containing the input for test data.
        y_test: A list containing the prediction labels of test data.
        thresh_percentage: An integer containing the threshold percentage.
        pred_window: An integer representing the size of prediction window.
        tails: A dictionary containing metric name and the tail to be considered.
        no_of_metrics: An integer representing number of metrics.
        components: An integer representing the number of gaussian components.
    Returns:
        df_probab_u: A dataframe containing the probability value.
        threshold_unscaled: A dictionary containing unscaled thresholds for each metric.
    """
    golden_metrics = []
    thresh_u=[]
    thresh_l=[]
    for i in list(mdn_layers):
        golden_metrics.append(i.split('_forecast')[0])
    thresh_percentage_throughput = 90
    pos_threshold_throughput = st.norm.ppf(thresh_percentage_throughput/100)
    pos_threshold = st.norm.ppf(thresh_percentage/100)
    for i in golden_metrics:
        if(i=='throughput'):
            data = input_and_labels[i].values
            thresh_u.append(pos_threshold_throughput*np.std(data)+np.mean(data))
            thresh_l.append(-pos_threshold_throughput*np.std(data)+np.mean(data))
        else:
            data = input_and_labels[i].values
            thresh_u.append(pos_threshold*np.std(data)+np.mean(data))
            thresh_l.append(-pos_threshold*np.std(data)+np.mean(data))
    
    threshold_unscaled ={}
    for i in range(len(golden_metrics)):
        if(tails[golden_metrics[i]] == 'l'):
            threshold_unscaled[golden_metrics[i]] =thresh_l[i]
        elif(tails[golden_metrics[i]] == 'r'):
            threshold_unscaled[golden_metrics[i]] =thresh_u[i]
    thresh_u=scalerY.transform([thresh_u])
    thresh_l=scalerY.transform([thresh_l])
    thresholds ={}
    for i in range(len(golden_metrics)):
        thresholds[golden_metrics[i]] = [thresh_l[0][i],thresh_u[0][i]]
    prediction_output=model.predict(testX) ##
    df_probab_u=pd.DataFrame()
    metrics = []
    prediction_output= prediction_output[no_of_metrics:] 
    for i in range(0,len(mdn_layers)):
        metrics.append(mdn_layers[i])
        alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(prediction_output[i])
        pred_val = 0
        for comp in range(components):
            pred_val += alpha_pred[:,comp]*mu_pred[:,comp]
        if(tails[golden_metrics[i]]=='r'):
            thresh_uu=thresholds[mdn_layers[i].split('_forecast')[0]][1]
            probabs_list_u=prob_calc_u(alpha_pred,mu_pred,sigma_pred,thresh_uu)
            probabs_s_u = pd.Series(probabs_list_u)
            df_probab_u[mdn_layers[i]+'_prob']=probabs_s_u
        elif(tails[golden_metrics[i]]=='l'):
            thresh_ll=thresholds[mdn_layers[i].split('_forecast')[0]][0]
            probabs_list_l=prob_calc_l(alpha_pred,mu_pred,sigma_pred,thresh_ll)
            probabs_s_l = pd.Series(probabs_list_l)
            df_probab_u[mdn_layers[i]+'_prob']=probabs_s_l
        df_probab_u[mdn_layers[i]+'_mdn_output'] = pred_val
    y_pred = []
    for col in df_probab_u.columns:
        if(col.endswith('_prob')):
            y_pred.append(df_probab_u[col].values)
    auc_scores = []
    y_label_pred= []
    data = []
    for metric in metrics:
        data.append([metric.split('_forecast')[0]])
    for i in range(len(mdn_layers)):
        tpr,fpr,thresholds = generate_roc(y_test[i],y_pred[i])
        auc_scores.append(auc(fpr,tpr))
        cutoff = generate_youden(tpr,fpr,thresholds)
        y_pred_ = np.where(y_pred[i]>cutoff,1,0)
        df_probab_u[mdn_layers[i]+'_mdn_alert'] = y_pred_
        cm=confusion_matrix(y_test[i],y_pred_)
        if(len(cm)==2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn=cm.ravel()[0]
            fp=0
            fn=0
            tp=0
        data[i].append(tn)
        data[i].append(fn)
        data[i].append(tp)
        data[i].append(fp)
    for auc_score in range(len(auc_scores)):
        if(data[auc_score][0]=='throughput' and thresh_percentage>95):
            data[auc_score].append(float('NaN'))
        else:
            data[auc_score].append(auc_scores[auc_score])
    df = pd.DataFrame(data,columns=['Golden_metric','True Negative','False Negative','True Positive','False Positive','AUC'])
    df.to_csv(tgt_dir + file_det + '_' + str(thresh_percentage) +'_MDN_auc.csv',index=False)
    return df_probab_u,threshold_unscaled

def generate_test_data(file_pred,classifier_layers,split_size,timesteps):
    """Generates test date for AUC evaluation.
    
    Args:
        file_pred: A string representing the file to be used for evaluation.
        classifier_layers: A list containing columns for classifiers.
        split_size: A float representing the fraction of train-test split.
        timesteps: An integer representing the timesteps taken by the prediction window.
    Returns:
        y_test: A list containing te prediction label of test data.
        test: A dataframe containing test data.
    """   
    df_pred = pd.read_csv(file_pred)
    if 'index' in df_pred.columns:
        df_pred.drop(columns=['index'],inplace=True)
    train, test = train_test_split(df_pred, test_size=split_size, random_state=42, shuffle=False)
    y_test=[]
    test.reset_index(drop=True,inplace=True)
    for metric_i in classifier_layers:
        y_test.append(test.loc[timesteps-1:,metric_i])
    return y_test,test

def generate_csv(test,df_prob,classifier_layers,mdn_layers,thresholds,tgt_dir,file_det,timesteps):
    """Pre-processes data and defines classifier and mdn output layers
    
    Args:.
        test: A dataframe representing the test data.
        df_prob: A dataframe representing the probability prediction.
        classifier_layers: A list containing columns for classifier.
        mdn_layers: A list containing columns for MDN.
        thresholds: A dict containing unscaled thresholds for metrics.
        tgt_dir: A string containing the directory where the results are to be stored.
        file_det: A string containing the destination where file is to be stored.
    Returns:
        None
    """
    test= test.loc[timesteps-1:]
    df_prob['timestamp'] = test['timestamp'].values
    df_results = df_prob.merge(test,on='timestamp',how='outer')
    test.drop(columns=classifier_layers,inplace=True)
    test.drop(columns=mdn_layers,inplace=True)
    df_prob = df_prob.merge(test,on='timestamp',how='outer')
    df_prob.to_csv(tgt_dir+file_det+'_Dashboard' + '.csv',index=False)
    df_results.to_csv(tgt_dir+file_det+'_Results' + '.csv',index=False)
    with open(tgt_dir+file_det+'_Results' +'.json', "w") as write_file:
        json.dump(thresholds, write_file)