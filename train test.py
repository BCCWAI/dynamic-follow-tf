import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU
from keras.layers.advanced_activations import ELU
import numpy as np
import random
from normalizer import norm
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn import preprocessing
#np.set_printoptions(threshold=np.inf)
'''from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(3)'''

class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10==0:
            plt.clf()
            dist=23
            x = [i+1 for i in range(41)]
            
            x=range(len(x_train))
            y = [i for i in y_train]
            y2 = [model.predict(np.asarray([i]))[0][0] for i in x_train]
            
            plt.plot(x,y,label='ground-truth')
            plt.plot(x,y2,label='prediction-{}'.format(epoch))
            plt.title("braking to a stop")
            plt.legend()
            #plt.plot(x,y2,label='epoch-{}'.format(epoch))
            plt.pause(.1)
            '''print("Looking good?")
            stop = input("[Y/n]: ")
            if stop.lower()=="y":
                model.stop_training = True'''

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf")

with open("data/x", "r") as f:
    x_train = json.load(f)

with open("data/y", "r") as f:
    y_train = json.load(f)

NORM = True
if NORM:
    v_ego, v_ego_scale = (norm([i[0] for i in x_train]))
    a_ego, a_ego_scale = (norm([i[1] for i in x_train]))
    v_lead, v_lead_scale = (norm([i[2] for i in x_train]))
    x_lead, x_lead_scale = (norm([i[3] for i in x_train]))
    a_lead, a_lead_scale = (norm([i[4] for i in x_train]))
    
    x_train = [[v_ego[idx], a_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]] for (idx, i) in enumerate(v_ego)]
    #x_train.append([v_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]])
    
    x_train = np.asarray(x_train)
    
    #y_train = np.asarray([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
    #y_train = np.asarray(y_train)
    #scaler = preprocessing.StandardScaler().fit(x_train)
    #x_train = scaler.transform(x_train)
else:
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

'''for idx,i in enumerate(y_train):
    if i < -.5 and x_train[idx][0] > 8.9:
        print(i)
        print(idx)
        print(x_train[idx])
        break'''

#opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.SGD()
#opt = keras.optimizers.RMSprop(0.001, decay=1e-6)

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
options=[[8, 128]]
#counter=0
for i in options:
    layer_num=i[0]
    nodes=i[1]
    a_function="relu"
    
    model = Sequential()
    model.add(Dense(5, input_shape=(x_train.shape[1:])))
    #model.add(Dropout(.1))
    for i in range(layer_num):
        model.add(Dense(nodes, activation=a_function))
        #model.add(Dropout(.01))
    model.add(Dense(1))
        
    
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
    #tensorboard = TensorBoard(log_dir="logs/{}-layers-{}-nodes-{}".format(layer_num, nodes, a_function))
    visualize = Visualize()
    model.fit(x_train, y_train, batch_size=12, epochs=41, callbacks=[visualize]) #callbacks=[tensorboard])
    
    #data = [norm(23.74811363, v_ego_scale), norm(-0.26912481, a_ego_scale), norm(15.10309029, v_lead_scale), norm(55.72000122, x_lead_scale), norm(-0.31268027, a_lead_scale)] #should be -0.5
    #prediction=model.predict(np.asarray([[norm(23.74811363, v_ego_scale), norm(15.10309029, v_lead_scale), norm(30.72000122, x_lead_scale)]]))[0][0]
    #print((prediction - 0.5)*2.0)
    
    #accur = list([list(i) for i in x_train])
    
    '''if NORM:
        dist=23
        x = [i+1 for i in range(41)]
        y = [model.predict(np.asarray([[norm(dist, v_ego_scale), norm(0, a_ego_scale), norm(dist, v_lead_scale), norm(i+1, x_lead_scale), norm(0, a_lead_scale)]]))[0][0] for i in range(41)]
        y2 = [model.predict(np.asarray([[norm(dist, v_ego_scale), norm(0, a_ego_scale), norm(i+1, v_lead_scale), norm(dist, x_lead_scale), norm(0, a_lead_scale)]]))[0][0] for i in range(41)]
        
        f1 = plt.figure(1)
        if counter==0:
            plt.plot([0, 40], [.5, .5], '--', linewidth=1)
            plt.plot([dist, dist], [.4, .6], '--', linewidth=1)
        plt.plot(x,y,label='{}-node-{}-layers'.format(nodes, layer_num))
        plt.title('distance')
        plt.legend()
        
        f2 = plt.figure(2)
        if counter==0:
            plt.plot([0, 40], [.5, .5], '--', linewidth=1)
            plt.plot([dist, dist], [.3, .7], '--', linewidth=1)
        plt.plot(x,y2,label='{}-node-{}-layers'.format(nodes, layer_num))
        plt.title('velocity')
        plt.legend()
        #plt.xlabel('m/s - m')
        #plt.ylabel('gas/brake percentage (0-1)')
        plt.pause(.1)
    else:
        y = [model.predict(np.asarray([[17.8816, 17.8816, i, 0]]))[0][0] for i in range(40)]
        x = [i for i in range(40)]
        plt.plot(x,y)
        plt.show()
    counter+=1'''
    
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
    model_name = "model3"
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)