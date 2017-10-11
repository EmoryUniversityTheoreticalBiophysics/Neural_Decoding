#Import standard packages

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import io
from scipy import stats
import pickle

#Import function to get the covariate matrix that includes spike history from previous bins
from preprocessing_funcs import get_spikes_with_history

#Import metrics
from metrics import get_R2
from metrics import get_rho

#Import decoder functions
from decoders import SimpleRNNDecoder
from decoders import GRUDecoder
from decoders import LSTMDecoder

folder='/home/david/Projects/Neuroscience/Precise Spike Timing/Neural_Decoding/' #ENTER THE FOLDER THAT YOUR DATA IS IN
# folder='/home/jglaser/Data/DecData/'
# folder='/Users/jig289/Dropbox/Public/Decoding_Data/'

with open(folder+'example_data_s1.pickle','rb') as f:
#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
    neural_data,vels_binned=pickle.load(f) #If using python 2

bins_before=6    #How many bins of neural data prior to the output are used for decoding
bins_current=1   #Whether to use concurrent time bin of neural data
bins_after=6     #How many bins of neural data after the output are used for decoding

# Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
# Function to get the covariate matrix that includes spike history from previous bins
X=get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)

#Set decoding output
y=vels_binned

#Set what part of data should be part of the training/testing/validation sets
training_range=[0, 0.7]
testing_range=[0.7, 0.85]
valid_range=[0.85, 1]

num_examples=X.shape[0]

#Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
#This makes it so that the different sets don't include overlapping neural data
training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,
                       np.int(np.round(training_range[1]*num_examples))-bins_after)
testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,
                      np.int(np.round(testing_range[1]*num_examples))-bins_after)
valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,
                    np.int(np.round(valid_range[1]*num_examples))-bins_after)

#Get training data
X_train=X[training_set, :, :]
y_train=y[training_set, :]

#Get testing data
X_test=X[testing_set, :, :]
y_test=y[testing_set, :]

#Get validation data
X_valid=X[valid_set, :, :]
y_valid=y[valid_set, :]

#Z-score "X" inputs.
X_train_mean=np.nanmean(X_train,axis=0)
X_train_std=np.nanstd(X_train,axis=0)
X_train=(X_train-X_train_mean)/X_train_std
X_test=(X_test-X_train_mean)/X_train_std
X_valid=(X_valid-X_train_mean)/X_train_std

#Zero-center outputs
y_train_mean = np.mean(y_train, axis=0)
y_train = y_train - y_train_mean
y_test = y_test - y_train_mean
y_valid = y_valid - y_train_mean

#Declare model
model_rnn = SimpleRNNDecoder(units = 400, dropout = 0, num_epochs = 5)

#Fit model
model_rnn.fit(X_train, y_train)

#Get predictions
y_valid_predicted_rnn = model_rnn.predict(X_valid)

#Get metric of fit
R2s_rnn = get_R2(y_valid, y_valid_predicted_rnn)
print('R2s:', R2s_rnn)
