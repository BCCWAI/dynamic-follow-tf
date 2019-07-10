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
#np.set_printoptions(threshold=np.inf)
'''from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(3)'''

'''gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))'''

class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # print model accuracy:
        accuracy = []
        for i in range(1000):
            choice = random.randint(0, len(x_train) - 2)
            real = y_train[choice]
            to_pred = x_train[choice]
            pred = model.predict(np.asarray([to_pred]))[0][0]
            accuracy.append(abs(real-pred))
            
        avg = sum(accuracy) / len(accuracy)
        print("Accuracy: {}".format(abs(avg-1)))
        
        if epoch % 1==0:
            dist=25
            x = [i+1 for i in range(41)]
            if NORM:
                y = [model.predict(np.asarray([[normX(dist, v_scale), normX(0, a_scale), normX(dist, v_scale), normX(i+1, x_scale), normX(0, a_scale)]]))[0][0] for i in range(41)]
                y2 = [model.predict(np.asarray([[normX(dist, v_scale), normX(0, a_scale), normX(i+1, v_scale), normX(dist, x_scale), normX(0, a_scale)]]))[0][0] for i in range(41)]
                #y = [model.predict(scaler.transform([[dist, 0, dist, i+1, 0]]))[0][0] for i in range(41)]
                #y2 = [model.predict(scaler.transform([[dist, 0, i+1, dist, 0]]))[0][0] for i in range(41)]
            else:
                y = [model.predict(np.asarray([[dist, 0, dist, i+1, 0]]))[0][0] for i in range(41)]
                y2 = [model.predict(np.asarray([[dist, 0, i+1, dist, 0]]))[0][0] for i in range(41)]
            
            f1 = plt.figure(1)
            plt.clf()
            #if epoch==0:
            if NORM:
                if NORM_Y:
                    plt.plot([0, 40], [.5, .5], '--', linewidth=1)
                    plt.plot([dist, dist], [.4, .6], '--', linewidth=1)
                else:
                    plt.plot([0, 40], [0.0, 0.0], '--', linewidth=1)
                    plt.plot([dist, dist], [-0.2, 0.2], '--', linewidth=1)
            else:
                plt.plot([0, 40], [0.0, 0.0], '--', linewidth=1)
                plt.plot([dist, dist], [-0.2, 0.2], '--', linewidth=1)
            #plt.plot(x,y,label='{}-node-{}-layers'.format(nodes, layer_num))
            plt.plot(x,y,label='epoch-{}'.format(epoch))
            plt.title('distance')
            plt.legend()
            plt.savefig("models/h5_models/{}/1-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.1)
            
            f2 = plt.figure(2)
            plt.clf()
            #if epoch==0:
            if NORM:
                if NORM_Y:
                    plt.plot([0, 40], [.5, .5], '--', linewidth=1)
                    plt.plot([dist, dist], [.3, .7], '--', linewidth=1)
                else:
                    plt.plot([0, 40], [0.0, 0.0], '--', linewidth=1)
                    plt.plot([dist, dist], [-0.4, 0.4], '--', linewidth=1)
            else:
                plt.plot([0, 40], [0.0, 0.0], '--', linewidth=1)
                plt.plot([dist, dist], [-.5, 0.5], '--', linewidth=1)
            #plt.plot(x,y2,label='{}-node-{}-layers'.format(nodes, layer_num))
            plt.plot(x,y2,label='epoch-{}'.format(epoch))
            plt.title('velocity')
            plt.legend()
            plt.savefig("models/h5_models/{}/2-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.1)
            
            f3 = plt.figure(3)
            plt.clf()
            x = [i+1 for i in range(41)]
            x=range(len(x_test))
            y = [i for i in y_test]
            if NORM:
                y2 = [model.predict(np.asarray([[normX(i[0], v_scale), normX(i[1], a_scale), normX(i[2], v_scale), normX(i[3], x_scale), normX(i[4], a_scale)]]))[0][0] for i in x_test]
            else:
                y2 = [model.predict(np.asarray([i]))[0][0] for i in x_test]
            
            plt.plot(x,y,label='ground-truth')
            plt.plot(x,y2,label='epoch-{}'.format(epoch))
            plt.title("braking to a stop")
            plt.legend()
            plt.savefig("models/h5_models/{}/3-{}-epoch-{}.png".format(model_name, model_name, epoch))
            #plt.xlabel('m/s - m')
            #plt.ylabel('gas/brake percentage (0-1)')
            plt.pause(.1)
            model.save("models/h5_models/{}/{}-epoch-{}.h5".format(model_name, model_name, epoch))
            '''if epoch - 1 % 2 == 0 and epoch!=0:
                print("Stop training?")
                stop = input("[Y/n]: ")
                if stop.lower()=="y":
                    model.stop_training = True'''

os.chdir("C:/Git/dynamic-follow-tf")
data_dir = "3model"
norm_dir = "data/{}/normalized"
model_name = "3model"

NORM_Y = True
new_data = False


with open("data/x", "r") as f:
    x_test = json.load(f)
with open("data/y", "r") as f:
    y_test = json.load(f)
    if NORM_Y:
        y_test = [np.interp(i, [-1, 1], [0, 1]) for i in y_test]


NORM = True
if NORM:
    if os.path.exists(norm_dir.format(data_dir)) and not new_data:
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
    
    v_scale, a_scale, x_scale = normalized['v_scale'], normalized['a_scale'], normalized['x_scale']
    
    x_train = np.asarray(normalized['normalized'])
    
    #x_train = [[v_ego[idx], a_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]] for (idx, i) in enumerate(v_ego)]
    #x_train.append([v_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]])
    if NORM_Y:
        y_train = np.asarray([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
    else:
        y_train = np.asarray(y_train)
else:
    print("Loading data...")
    with open("data/{}/x_train".format(data_dir), "rb") as f:
        x_train = pickle.load(f)
    with open("data/{}/y_train".format(data_dir), "rb") as f:
        y_train = pickle.load(f)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

def soft_acc(y_true, y_pred):
    return 1 - K.abs(y_true - y_pred)

#opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.001)

'''model = Sequential([
    Dense(5, activation="tanh", input_shape=(x_train.shape[1:])),
    Dense(8, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(32, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(128, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(32, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(8, activation="tanh"),
    Dense(1),
  ])'''
#[12, 324]
options=[[5, 2048]] # good ones: [[8, 1000], [7, 2500], [4, 2048], [4, 4096]], best so far: [[3, 8096], [2, 8096]] (adadelta)

for i in options:
    layer_num=i[0] - 1
    nodes=i[1]
    a_function="relu"
    
    model = Sequential()
    model.add(Dense(nodes, activation=a_function, input_shape=(x_train.shape[1:])))
    #model.add(Dropout(.1))
    for i in range(layer_num):
        model.add(Dense(nodes, activation=a_function))
        #model.add(Dropout(.2))
    model.add(Dense(1))
        
    
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    #tensorboard = TensorBoard(log_dir="logs/{}-layers-{}-nodes-{}".format(layer_num, nodes, a_function))
    visualize = Visualize()
    model.fit(x_train, y_train, batch_size=2048, epochs=50, callbacks=[visualize]) #callbacks=[tensorboard])
    
    #data = [norm(23.74811363, v_ego_scale), norm(-0.26912481, a_ego_scale), norm(15.10309029, v_lead_scale), norm(55.72000122, x_lead_scale), norm(-0.31268027, a_lead_scale)] #should be -0.5
    #prediction=model.predict(np.asarray([[norm(23.74811363, v_ego_scale), norm(15.10309029, v_lead_scale), norm(30.72000122, x_lead_scale)]]))[0][0]
    #print((prediction - 0.5)*2.0)
    
    #accur = list([list(i) for i in x_train])
    
    
'''
accuracy=[]
for i in range(500):
    choice = random.randint(0, len(x_train) - 1)
    real=y_train[choice]
    to_pred = list(list(x_train)[choice])
    pred = model.predict(np.asarray([to_pred]))[0][0]
    accuracy.append(abs(real-pred))
    #print("Real: "+str(real))
    #print("Prediction: "+str(pred))
    #print()
    
avg = sum(accuracy) / len(accuracy)
if NORM:
    print("Accuracy: "+ str(abs(avg-1)))
else:
    print("Accuracy: "+ str(np.interp(avg, [0, 1], [1, 0])))
print()
print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5
'''
#test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

#print(model.predict(np.asarray(test_data)))

save_model = False
tf_lite = False
if save_model:
    model_name = "holden"
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)