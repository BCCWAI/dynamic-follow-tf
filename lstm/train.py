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

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
        plt.savefig("models/h5_models/{}/0-{}-epoch-{}.png".format(model_name, model_name, epoch))
        plt.pause(.1)
        model.save("models/h5_models/{}/{}-epoch-{}.h5".format(model_name, model_name, epoch))
        '''if epoch % 20==0 and epoch!=0:
            print("Stop training?")
            stop = input("[Y/n]: ")
            if stop.lower()=="y":
                model.stop_training = True'''

#v_ego, v_lead, d_lead
data_dir = "LSTM_fake"
os.chdir("C:/Git/dynamic-follow-tf")

norm_dir = "data/{}/normalized"

model_name = "LSTM_fake"

with open("data/{}/x_train".format(data_dir), "rb") as f:
    x_train = pickle.load(f)

with open("data/{}/y_train".format(data_dir), "rb") as f:
    y_train = pickle.load(f)

NORM = True
if NORM:
    if not os.path.exists(norm_dir.format(data_dir)):
        print("Normalizing...")
        sys.stdout.flush()
        normalized = norm(x_train)
        #normalized['normalized'] = np.array(normalized['normalized'])
        print("Dumping normalization...")
        with open(norm_dir.format(data_dir), "wb") as f:
            pickle.dump(normalized, f)
    else:
        print("Loading normalized data...")
        with open(norm_dir.format(data_dir), "rb") as f:
            normalized = pickle.load(f)
    
    #v_scale, a_scale, x_scale = normalized['v_scale'], normalized['a_scale'], normalized['x_scale']
    scales = normalized['scales']
    x_train = normalized['normalized']
    #x_train = norm(x_train)
    y_train = np.array([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
else:
    y_train = np.array([np.interp(i, [-1, 1], [0, 1]) for i in y_train])

flatten = True
if flatten:
    x_train = np.array([[inner for outer in sample for inner in outer] for sample in x_train])
print(x_train.shape)

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

opt = keras.optimizers.Adam(lr=0.00003)#, decay=1e-6)
#opt = keras.optimizers.Adadelta(lr=0.01)
#opt = keras.optimizers.RMSprop(0.006)
#opt = keras.optimizers.Adagrad(lr=0.00001)
opt = 'adam'

layers = 6
nodes = 512

model = Sequential()
#model.add(CuDNNLSTM(16, input_shape=(x_train.shape[1:]), return_sequences=True))
#model.add(CuDNNLSTM(16))
model.add(Dense(nodes, activation="relu", input_shape=(x_train.shape[1:])))
#model.add(Dropout(.05))
for i in range(layers-1):
    model.add(Dense(nodes, activation="relu"))
    #model.add(Dropout(.05))
    #model.add(LSTM(64, input_shape=(x_train.shape[1:]), return_sequences=True))
#model.add(Permute((2,1), input_shape=(10, 5)))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
#tensorboard = TensorBoard(log_dir="logs/test-{}".format("30epoch"))
callback_list = [Visualize()]
model.fit(x_train, y_train, shuffle=True, batch_size=512, epochs=100, callbacks=callback_list)

'''for i in range(10):
    rand = random.randint(0, len(x_train))
    pred = model.predict([[x_train[rand]]])[0][0]
    print("Ground truth: {}".format(y_train[rand]))
    print("Prediction: {}\n".format(pred))'''

save_model = False
if save_model:
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")