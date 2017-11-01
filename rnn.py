#Import standard packages

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
# %matplotlib inline
from scipy import io
from scipy import stats
import pickle
from time import time
from sklearn.cross_validation import KFold

#Import function to get the covariate matrix that includes spike history from previous bins
from preprocessing_funcs import get_spikes_with_history

#Import metrics
from metrics import get_R2
from metrics import get_rho

#Import decoder functions
from decoders import SimpleRNNDecoder
from decoders import GRUDecoder
from decoders import LSTMDecoder

folder=''
# folder='/home/jglaser/Data/DecData/'
# folder='/Users/jig289/Dropbox/Public/Decoding_Data/'
t0 = time()

with open(folder+'example_data_s1.pickle','rb') as f:
#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
    neural_data,vels_binned,dt_ratio=pickle.load(f) #If using python 2
print(dt_ratio)
bins_before=3*dt_ratio    #How many bins of neural data prior to the output are used for decoding
bins_current=1   #Whether to use concurrent time bin of neural data
bins_after=3*dt_ratio      #How many bins of neural data after the output are used for decoding

# Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
# Function to get the covariate matrix that includes spike history from previous bins
inputdata = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current, dt_ratio)

fractions_of_data = np.asarray([1, 0.8, 0.6, 0.4, 0.2, 0.05])
fractions_of_data = np.asarray([1, 0.8])
crossval = 2

#Z-score "X" inputs.
inputdata_mean = np.nanmean(inputdata, axis=0)
inputdata_std = np.nanstd(inputdata, axis=0)
inputdata = (inputdata - inputdata_mean) / inputdata_std

#Zero-center outputs
vels_mean = np.mean(vels_binned, axis=0)
vels_binned = vels_binned - vels_mean

#Declare model
model_rnn = SimpleRNNDecoder(units = 400, dropout = 0, num_epochs = 5)
R2s_tmp = np.zeros((len(fractions_of_data),10,2))
for i, frac in enumerate(fractions_of_data):
    num_examples = int(inputdata.shape[0] * frac)
    X = inputdata[:num_examples, :, :]
    y = vels_binned[:num_examples, :]
    kf = KFold(num_examples, crossval)
    for j, (train, test) in enumerate(kf): 
        train = train[bins_before:-bins_after]
        test = test[bins_before:-bins_after]
        #Get training data
        X_train=X[train, :, :]
        y_train=y[train, :]

        #Get testing data
        X_test=X[test, :, :]
        y_test=y[test, :]

        #Fit model
        model_rnn.fit(X_train, y_train)

        #Get predictions
        y_test_predicted_rnn = model_rnn.predict(X_test)
 
        #Get metric of fit
        R2s_tmp[i, j, :] = get_R2(y_test, y_test_predicted_rnn)
        print('R2s:', R2s_tmp[i, j, :])

plt.errorbar(fractions_of_data**-1, np.mean(R2s_tmp[:,:,1], axis=1), np.std(R2s_tmp[:,:,1], axis=1))

plt.savefig("perf_over_data_dt" + str(60 * dt_ratio) + ".eps")

##Declare model
#model_lstm=LSTMDecoder(units=400,dropout=0,num_epochs=5)
#
##Fit model
#model_lstm.fit(X_train,y_train)
#
##Get predictions
#y_valid_predicted_lstm = model_lstm.predict(X_valid)
#
##Get metric of fit
#R2s_lstm = get_R2(y_valid,y_valid_predicted_lstm)
#print('R2s:', R2s_lstm)

print('elapsed time:', time()-t0)

