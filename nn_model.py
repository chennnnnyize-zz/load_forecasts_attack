#build the NN models: RNN module, NN module
import tensorflow
from tensorflow.python.ops import control_flow_ops
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import LSTM, Embedding,SimpleRNN
from keras.utils import np_utils
from tensorflow.python.platform import flags
from numpy import shape
import numpy as np
#from skimage import io, color, exposure, transform
import os
import glob
import h5py
import pandas as pd
import numpy as np


def rnn_model(seq_length, input_dim, output_dim):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(seq_length, input_dim), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    #model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    #model.add(Activation('relu'))
    model.add(Dense(32))
    #model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim, init='normal'))
    model.add(Activation('linear'))
    return model

def nn_model(input_dim, output_dim):
    model=Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim, init='normal'))
    model.add(Activation('linear'))

    return model


def lstm_model(seq_length, input_dim, output_dim):
    model = Sequential()
    #model.add(SimpleRNN(64, input_shape=(seq_length, input_dim), return_sequences=False))
    model.add(LSTM(64, input_shape=(seq_length, input_dim), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    #model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    #model.add(Activation('relu'))
    model.add(Dense(32))
    #model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim, init='normal'))
    model.add(Activation('linear'))
    return model


def svm_model(seq_length, input_dim):

    return model