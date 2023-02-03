"""Model Generation Module

This module allows the user to create the baseline model and trains it over the test
data.

This module contains the following functions:

    * model_builder: builds the model for given lstm variant.
    * model_fit: fits the model for given training data.
"""
import keras
from tensorflow.keras.layers import Input, LSTM, Dense,Concatenate,Dropout, LSTM,GRU,SimpleRNN
from tensorflow.keras.layers import Bidirectional
from keras import regularizers
from tensorflow.keras.models import Model
from operations import exp,gnll_loss


def model_builder(trainX,mdn_layers,lstm_variant,neurons,components):
    """Builds the model for given lstm variant.
    
    Args:.
        trainX: A numpy array containing input of train data.
        classifier_layers: A list containing columns for classifier.
        mdn_layers: A list containing columns for MDN.
        lstm_variant: A string representing type of temporal layer to be used.
        neurons: An integer representing number of neurons in MDN layer.
        components: An integer representing number of gaussian distribution used.
    Returns:
        model: A Tensorflow Model build using the proposed architecture.
    Raises:
        TypeError: Raises a typeerror if Invalid temporal model is given as input.
    """
    input_1 = Input(shape=(trainX.shape[1], trainX.shape[2]), name='input_1')
    if(lstm_variant=='bilstm'):
        lstm_1 = Bidirectional(LSTM(128,name='lstm_1',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))(input_1)
        d2=Dropout(0.2,name="drop2")(lstm_1)
    
    elif(lstm_variant=='stacked_bilstm'):
        lstm_1 = Bidirectional(LSTM(128,name='lstm_1',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),return_sequences=True))(input_1)
        d1=Dropout(0.2,name="drop1")(lstm_1)
        lstm_2 = Bidirectional(LSTM(128,name='lstm_2',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))(d1)
        d2=Dropout(0.2,name="drop2")(lstm_2)

    
    elif(lstm_variant=='lstm'):
        lstm_1 = LSTM(128,name='lstm_1',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(input_1)
        d2=Dropout(0.2,name="drop2")(lstm_1)
    
    
    elif(lstm_variant=='stacked_lstm'):
        lstm_1 = LSTM(128,name='lstm_1',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),return_sequences=True)(input_1)
        d1=Dropout(0.2,name="drop1")(lstm_1)
        lstm_2 = LSTM(128,name='lstm_2',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(d1)
        d2=Dropout(0.2,name="drop2")(lstm_2)
        
    elif(lstm_variant=='gru'):
        gru_1 = GRU(128,name='gru_1',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(input_1)
        d2=Dropout(0.2,name="drop2")(gru_1)

    elif(lstm_variant=='simpleRNN'):
        gru_1 = SimpleRNN(128,name='simpleRNN_1',
                                   bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(input_1)
        d2=Dropout(0.2,name="drop2")(gru_1)

    else:
        print('Invalid type: ',lstm_variant)
        raise TypeError("Only simpleRNN/gru/stacked_lstm/lstm/stacked_bilstm/bilstm supported currently.")
    
    category_output=[]
    metrics_array = {}
    loss_array = {}
    for i, dense_layer in enumerate(mdn_layers):
        name_0 = f'output_{dense_layer}' 
        name_1 = f'dense_1_{dense_layer}'
        name_2 = f'dense_2_{dense_layer}'
        h1 = Dense(neurons, activation="relu", name=name_1+"_mdn")(d2)
        h2 = Dense(neurons, activation="relu", name=name_2+"_mdn")(h1)
        alphas = Dense(components, activation="softmax", name=name_0+"_alphas")(h2)
        mus = Dense(components, name=name_0+"_mus")(h2)
        sigmas = Dense(components, activation=exp, name=name_0+"_sigmas")(h2)
        pvec = Concatenate(name=name_0)([alphas,mus,sigmas])
        category_output.append(pvec)
        loss_array[name_0] =gnll_loss
    model = Model(inputs=input_1, outputs=category_output)
    model.compile(optimizer='adam',
                  loss=loss_array,
                  metrics=metrics_array)    
    return model

def model_fit(model,trainX,testX,y_train_mdn,y_test_mdn,epoch_cnt,batch_size): 
    """Fits the given model.
    
    Args:
        model: A Tensorflow Model build using the proposed architecture.
        trainX: A numpy array containing input of train data.
        testX: A numpy array containing input of test data.
        y_train_mdn: A list containing mdn output of train data.
        y_test_mdn: A list containing mdn output of test data.
        epoch_cnt: An integer representing the number of epochs to run.
        batch_size: An integer representing the batch size.
    Returns:
        history: An object representing the history of performance during evaluation.
    """
    history=model.fit(trainX,y_train_mdn,validation_data=(testX,y_test_mdn), epochs=epoch_cnt, batch_size=batch_size,verbose=1)
    return history