import os
import json
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, BatchNormalization, LeakyReLU, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Reshape, RNN, Permute
import numpy as np
import random
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
import time
from normalizer import norm
import matplotlib.pyplot as plt
import pickle
import sys
from sklearn import preprocessing
import itertools

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        preds = []
        grounds = []
        x = list(range(len(random_choices)))
        for i in random_choices:
            preds.append(model.predict([[x_train[i]]])[0][0])
            grounds.append(y_train[i])
        
        plt.clf()
        plt.plot(x, preds, label='prediction-{}'.format(epoch))#, marker='o')
        plt.plot(x, grounds, label='ground-truth')#, marker='o')
        plt.title("random samples")
        plt.legend()
        plt.pause(.1)
        if epoch % 20==0 and epoch!=0:
            print("Stop training?")
            stop = input("[Y/n]: ")
            if stop.lower()=="y":
                model.stop_training = True

#v_ego, v_lead, d_lead
data_dir="LSTM_new"
os.chdir("C:/Git/dynamic-follow-tf")

with open("data/{}/x_train".format(data_dir), "rb") as f:
    x_train = pickle.load(f)
    x_train = [[inner for outer in sample for inner in outer] for sample in x_train]
    x_train = np.array(x_train)
with open("data/{}/y_train".format(data_dir), "rb") as f:
    y_train = pickle.load(f)


NORM = True
if NORM:
    print("Normalizing...")
    sys.stdout.flush()
    #normalized = norm(x_train)
    #v_scale, a_scale, x_scale = normalized['v_scale'], normalized['a_scale'], normalized['x_scale']
    #x_train = np.asarray(normalized['normalized'])
    #x_train = norm(x_train)
    y_train = np.array([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
    

random_choices = []
for i in y_train:
    rand=random.randint(0, len(y_train)-1)
    if y_train[rand] > 0.7: # 0.3
        random_choices.append(rand)
    if len(random_choices) == 100:
        break
for i in y_train:
    rand=random.randint(0, len(y_train)-1)
    if abs(y_train[rand]) < 0.5 and y_train[rand]!=0.5:
        random_choices.append(rand)
    if len(random_choices) == 200:
        break
for i in y_train:
    rand=random.randint(0, len(y_train)-1)
    if y_train[rand] < 0.7:
        random_choices.append(rand)
    if len(random_choices) == 300:
        break

#random_choices = [random.randint(0, len(y_train)) for i in range(50)]
random_choices = [x for _,x in sorted(zip([y_train[i] for i in random_choices], random_choices))] # sort from lowest to highest for visualization

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
#opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.RMSprop(0.0000001)

layers=10
nodes=3450

model = Sequential()
#model.add(Flatten())
model.add(Dense(20, activation="relu", input_shape=(x_train.shape[1:])))
for i in range(layers):
    model.add(Dense(nodes, activation="relu"))
    #model.add(Dropout(.1))
#model.add(LSTM(64, input_shape=(x_train.shape[1:])))#, return_sequences=True))
#model.add(Permute((2,1), input_shape=(10, 5)))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
#tensorboard = TensorBoard(log_dir="logs/test-{}".format("30epoch"))
callback_list = [Visualize()]
model.fit(x_train, y_train, shuffle=True, batch_size=5000, epochs=50, callbacks=callback_list)

'''for i in range(10):
    rand = random.randint(0, len(x_train))
    pred = model.predict([[x_train[rand]]])[0][0]
    print("Ground truth: {}".format(y_train[rand]))
    print("Prediction: {}\n".format(pred))'''

save_model = True
if save_model:
    model_name = "LSTM"
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")