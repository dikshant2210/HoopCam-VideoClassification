import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle as pkl
from baseline.model import lstm_model


batch_size = 32
epochs = 10
actions_list = ['JS', 'D', 'C', 'L', 'BTL', 'BTB', 'SM', 'HS', 'None']

index = dict()
for value, key in enumerate(actions_list):
    index[key] = value

df = pd.read_csv('baseline/hoop.csv')
files = df['Name'].values
target_vectors = df['ActionsFound'].values

x_train, x_valid, y_train, y_valid = train_test_split(files, target_vectors, test_size=0.2)
path_to_store = os.path.join(os.getcwd(), 'input', 'Inception_Activations')


def train_generator():
    while True:
        for start in range(0, len(x_train), batch_size):
            x_batch = list()
            y_batch = list()
            end = min(start+batch_size, len(x_train))
            train_batch = x_train[start:end]
            target_batch = y_train[start:end]
            for name, actions in zip(train_batch, target_batch):
                y = np.zeros(len(actions_list))
                path_to_file = os.path.join(path_to_store, name)
                with open(path_to_file, 'rb') as file:
                    array = pkl.load(file)
                x_batch.append(array)
                for action in actions.split('|'):
                    y[index[action]] = 1
                y_batch.append(y)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(x_valid), batch_size):
            x_batch = list()
            y_batch = list()
            end = min(start+batch_size, len(x_valid))
            valid_batch = x_valid[start:end]
            target_batch = y_train[start:end]
            for name, actions in zip(valid_batch, target_batch):
                y = np.zeros(len(actions_list))
                path_to_file = os.path.join(path_to_store, name)
                with open(path_to_file, 'rb') as file:
                    array = pkl.load(file)
                x_batch.append(array)
                for action in actions.split('|'):
                    y[index[action]] = 1
                y_batch.append(y)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


model = lstm_model()

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(x_train)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(x_valid)) / float(batch_size)))

model.save_weights('baseline/weights/model.h5')
