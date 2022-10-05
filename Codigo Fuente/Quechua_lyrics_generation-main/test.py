from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import sys
import io
import os
import codecs

SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 3
STEP = 1
BATCH_SIZE = 32

# load file model
filename = "saved_model.pb"

def get_model(dropout=0.2):
    print('Construyendo Modelo...')
    model = Sequential() #
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model

model = get_model()

model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
