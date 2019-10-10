import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU
from keras.layers.advanced_activations import ELU
from keras.activations import selu
import numpy as np
import random
from normalizer import normX
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
from keras import backend as K
import shutil
from sklearn.model_selection import train_test_split
import seaborn as sns

#np.set_printoptions(threshold=np.inf)
'''from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(3)'''

'''gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))'''
#gpu_options = tf.GPUOptions(allow_growth=True)
#session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            accuracy = []
            for i in range(500):
                choice = random.randint(0, len(x_test) - 2)
                real = y_test[choice]
                to_pred = x_test[choice]
                pred = model.predict([[to_pred]])[0][0]
                accuracy.append(abs(real-pred))
                
            avg = sum(accuracy) / len(accuracy)
            print("Accuracy: {}".format(abs(avg-1)))
            
            dist = 22.352
            graph_len = 40
            
            x = [i for i in range(graph_len + 1)]
            
            # --- DISTANCE ---
            plt.figure(1)
            plt.clf()
            
            dist_y = [model.predict(np.asarray([[normX(dist, scales['v_ego_scale']), normX(dist, scales['v_lead_scale']), normX(i, scales['x_lead_scale']), normX(0, scales['a_lead_scale'])]]))[0][0] for i in range(graph_len + 1)]
            
            plt.plot([0, graph_len], [.5, .5], '--', linewidth=1)
            plt.plot([dist, dist], [.4, .6], '--', linewidth=1)
        
            plt.plot(x, dist_y, label='epoch-{}'.format(epoch))
            plt.title('distance')
            plt.legend()
            plt.savefig("models/h5_models/{}/1-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.1)
            
            # --- VELOCITY ---
            plt.figure(2)
            plt.clf()
            
            vel_y = [model.predict(np.asarray([[normX(dist, scales['v_ego_scale']), normX(i, scales['v_lead_scale']), normX(dist, scales['x_lead_scale']), normX(0, scales['a_lead_scale'])]]))[0][0] for i in range(graph_len + 1)]

            plt.plot([0, graph_len], [.5, .5], '--', linewidth=1)
            plt.plot([dist, dist], [.3, .7], '--', linewidth=1)

            plt.plot(x, vel_y, label='epoch-{}'.format(epoch))
            plt.title('velocity')
            plt.legend()
            plt.savefig("models/h5_models/{}/2-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.1)
            
            # --- GROUND TRUTH ---
            plt.figure(3)
            plt.clf()
            
            pred_num = 100
            
            x = range(pred_num)
            ground_y = [i for i in y_test[:pred_num]]

            pred_y = [model.predict(np.asarray([i]))[0][0] for i in x_test[:pred_num]]
            
            plt.plot(x, ground_y, label='ground-truth')
            plt.plot(x, pred_y, label='epoch-{}'.format(epoch))
            plt.title("ground truths")
            plt.legend()
            plt.savefig("models/h5_models/{}/3-{}-epoch-{}.png".format(model_name, model_name, epoch))

            plt.pause(.1)
            model.save("models/h5_models/{}/{}-epoch-{}.h5".format(model_name, model_name, epoch))
            '''if epoch - 1 % 2 == 0 and epoch!=0:
                print("Stop training?")
                stop = input("[Y/n]: ")
                if stop.lower()=="y":
                    model.stop_training = True'''

os.chdir("C:/Git/dynamic-follow-tf")
data_dir = "gm-only"
norm_dir = "data/{}/normalized"
model_name = "gm-only"

try:
    shutil.rmtree("models/h5_models/{}".format(model_name))
except:
    pass

'''with open("data/x", "r") as f:
    x_test = json.load(f)
with open("data/y", "r") as f:
    y_test = json.load(f)
    y_test = [np.interp(i, [-1, 1], [0, 1]) for i in y_test]'''


if os.path.exists(norm_dir.format(data_dir)):
    print("Loading normalized data...")
    with open(norm_dir.format(data_dir), "rb") as f:
        normalized = pickle.load(f)
    
    with open("data/{}/y_train".format(data_dir), "rb") as f: # still have to load y_train, not stored as norm
        y_train = pickle.load(f)
else:
    print("Loading data...")
    with open("data/{}/x_train".format(data_dir), "rb") as f:
        x_train = pickle.load(f)
    with open("data/{}/y_train".format(data_dir), "rb") as f:
        y_train = pickle.load(f)
    
    print("Normalizing data...")
    normalized = normX(x_train)
    
    print("Dumping normalized data...")
    with open(norm_dir.format(data_dir), "wb") as f:
        pickle.dump(normalized, f)

scales = normalized['scales']

x_train = normalized['normalized']

y_train = np.array([np.interp(i, [-1, 1], [0, 1]) for i in y_train])

plt.clf()
sns.set_style('whitegrid')
sns.kdeplot([i[0] for i in x_train], bw=0.01)
plt.show()

'''

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

try:
    os.mkdir("models/h5_models/{}".format(model_name))
except:
    pass

opt = keras.optimizers.Adam(lr=0.0001)
#opt = keras.optimizers.Adadelta(lr=.000375)
#opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.00005)#, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.00025)
#opt = keras.optimizers.Adagrad(lr=0.001)
#opt = 'adam'

#opt = 'rmsprop'
#opt = keras.optimizers.Adadelta()

options=[[4, 64]] # good ones: [[8, 1000], [7, 2500], [4, 2048], [4, 4096]], best so far: [[3, 8096], [2, 8096]] (adadelta)

for i in options:
    layer_num=i[0] - 1
    nodes=i[1]
    a_function = "relu"
    
    model = Sequential()
    model.add(Dense(nodes, activation=a_function, input_shape=(x_train.shape[1:])))
    
    for i in range(layer_num):
        model.add(Dense(nodes, activation=a_function))
    model.add(Dense(1, activation='linear'))
        
    
    model.compile(loss='mean_squared_error', optimizer=opt)
    #tensorboard = TensorBoard(log_dir="logs/{}-layers-{}-nodes-{}".format(layer_num, nodes, a_function))
    visualize = Visualize()
    model.fit(x_train, y_train, shuffle=True, batch_size=512, epochs=50, validation_data=(x_test, y_test), callbacks=[visualize]) #callbacks=[tensorboard])

#print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5

save_model = False
tf_lite = False
if save_model:
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)'''