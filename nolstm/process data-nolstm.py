import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv

os.chdir("C:/Git/dynamic-follow-tf/data")
'''with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")'''

data_dir = "D:\Resilio Sync\df"
d_data = []
gm_counter = 0
other_counter = 0

CHEVY = True
REMOVE_COAST_CHEVY=False

HONDA = False
HOLDEN = True
MINSIZE = 60000 #kb #40000
print("Loading data...")
for folder in os.listdir(data_dir):
    if any([sup_car in folder for sup_car in ["CHEVROLET VOLT PREMIER 2017"]]) and CHEVY:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > MINSIZE: #if bigger than 40kb
                #print(os.path.join(os.path.join(data_dir, folder), filename))
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                
                started = False
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    line[0] = max(round(line[0], 12), 0.0)
                    if not started and line[0] != 0.0:
                        started = True
                    if started:
                        if REMOVE_COAST_CHEVY and line[-2] + line[-3] == 0.0 and "gbergman" not in folder: # skip coasting samples since has regen
                            continue
                        line[1] = round(line[1], 12) #a_ego
                        line[2] = max(round(line[2], 12), 0.0)
                        line[3] = max(round(line[3], 12), 0.0)
                        line[4] = round(line[4], 12) #a_lead
                        gm_counter+=1
                        d_data.append(line)
    
    elif any([sup_car in folder for sup_car in ["HOLDEN ASTRA RS-V BK 2017"]]) and HOLDEN:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > MINSIZE: #if bigger than 40kb
                #print(os.path.join(os.path.join(data_dir, folder), filename))
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                
                started = False
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    line[0] = max(round(line[0], 12), 0.0)
                    if not started and line[0] != 0.0:
                        started = True
                    if started:
                        line[1] = round(line[1], 12) #a_ego
                        line[2] = max(round(line[2], 12), 0.0)
                        line[3] = max(round(line[3], 12), 0.0)
                        line[4] = round(line[4], 12) #a_lead
                        gm_counter+=1
                        d_data.append(line)
    
    elif any([sup_car in folder for sup_car in ["HONDA CIVIC 2016 TOURING"]]) and HONDA:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > MINSIZE: #if bigger than 40kb
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                
                started = False
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    line[0] = max(round(line[0], 12), 0.0)
                    if not started and line[0] != 0.0:
                        started = True
                    if started:
                        line[1] = round(line[1], 12) #a_ego
                        line[2] = max(round(line[2], 12), 0.0)
                        line[3] = max(round(line[3], 12), 0.0)
                        line[4] = round(line[4], 12) #a_lead
                        if line[-2] > 1.0 and line[0] < .01: # user brake skyrockets when car is stopped for some reason, though since we have data from gm, we can exclude this data from honda
                            pass
                        else:
                            line[-2] = np.clip(line[-2], 0.0, 1.0) # sometimes goes neg when really no brake
                            d_data.append(line)
                            other_counter+=1
                        
    
    # the following should improve performance for deciding when and how much to apply gas (but might reduce braking performance)
    '''elif any([sup_car in folder for sup_car in ["TOYOTA COROLLA 2017", "TOYOTA PRIUS 2017", "TOYOTA RAV4 HYBRID 2017", "TOYOTA RAV4 2017"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > 40000: #if bigger than 40kb
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                for line in df:
                    if line != "" and "[" in line and "]" in line and len(line) >= 40:
                        line = ast.literal_eval(line)
                        line[6] = 0.0  # don't include brake pressure
                        d_data.append(line)
                        other_counter+=1
                        #if line[6] == 0.0 or line[5] > 0.0:  # for cars without brake sensor (like toyotas), only include lines with no brake. brake pressure is too inaccurate
                            #line[6] = 0.0  # don't include brake pressure
                            #other_counter+=1
                            #d_data.append(line)  # need to experiment with including braking samples, but setting brake to 0 so the model will coast instead of not knowing what to do'''

print("Filtering outliers...")        
driving_data = []
for line in d_data:  # do filtering
    if line[0] < -0.22352 or sum(line) == 0: #or (sum(line[:3]) == 0):
        continue
    if line[4] > 15 or line[4] < -15: # filter out crazy lead acceleration
        continue
    #line[0] = max(line[0], 0)
    #line[2] = max(line[2], 0)
    #line[3] = max(line[3], 0)
    
    #line[-1] = line[-1] / 4047.0  # only for corolla
    #line = [line[0], line[1], (line[2]-line[0]), line[3], line[4], line[5], line[6], line[7]] # this makes v_lead, v_rel instead
    driving_data.append(line)
random.shuffle(driving_data)

#dup_counter=0
#tmp_data=[i[:5] for i in driving_data]
#uniq = [i for i in tmp_data if tmp_data.count(i)>1]


even_out_vel = True
if even_out_vel: # evens out data based on v_ego
    print("Evening out vels...")
    #driving_data.sort(key = lambda x: x[0]) # sorts based on v_ego from smallest to largest
    max_vel=max([i[0] for i in driving_data])
    velocity_split=[[] for i in range(10)]
    vel_scales=[(max_vel/10)*(i+1) for i in range(10)]
    for line in driving_data:        
        location = [idx if line[0]>i else None for idx,i in enumerate(vel_scales)][::-1]
        if (len(set(location)) <= 1): # if below 4.105 m/s
            location=0
        else:
            location = max([i for i in location if i!=None])+1
        velocity_split[location].append(line)
    
    mod_val=10 #only remove data from first 5 vel sections
    to_modify=velocity_split[:mod_val]
    dont_modify=velocity_split[mod_val:]
    
    modified=[[]]*mod_val
    
    even_out = True # evens out based on gas
    if even_out:
        for idx, section in enumerate(to_modify):
            gas = [i for i in section if i[-3] - i[-2] > 0]
            nothing = [i for i in section if i[-3] - i[-2] == 0]
            brake = [i for i in section if i[-3] - i[-2] < 0]
            to_remove_gas = len(gas) - min(len(gas), len(nothing), len(brake)) if len(gas) != min(len(gas), len(nothing), len(brake)) else 0
            to_remove_nothing = len(nothing) - min(len(gas), len(nothing), len(brake)) if len(nothing) != min(len(gas), len(nothing), len(brake)) else 0
            to_remove_brake = len(brake) - min(len(gas), len(nothing), len(brake)) if len(brake) != min(len(gas), len(nothing), len(brake)) else 0
            del gas[:to_remove_gas]
            del nothing[:to_remove_nothing]
            del brake[:to_remove_brake]
            modified[idx] = gas + brake + nothing
            '''if idx==4:
                print(section[54])
                print(gas)
                print(brake)
                print(to_remove_gas)
                print(to_remove_brake)'''
    else:
        for idx, section in enumerate(to_modify):
            to_remove=len(to_modify[-1])
            for idi, sample in enumerate(section):
                if idi >= to_remove:
                    break
                modified[idx].append(sample)
    
    
    print("Before: {}".format([len(i) for i in velocity_split]))
    velocity_split=modified+dont_modify
    print("After: {}".format([len(i) for i in velocity_split]))
    driving_data=[]
    for section in velocity_split:
        for sample in section:
            driving_data.append(sample)
    #speeds=[i[0] for i in driving_data]
    #x=range(len(speeds))
    #plt.hist(x,speeds)
    #plt.show()


even_out_gas=False
if even_out_gas:  # makes number of gas/brake/nothing samples equal to min num of samples
    print("Evening out gas...")
    gas = [i for i in driving_data if i[-3] - i[-2] > 0]
    nothing = [i for i in driving_data if i[-3] - i[-2] == 0]
    brake = [i for i in driving_data if i[-3] - i[-2] < 0]
    to_remove_gas = 1303221#len(gas) - min(len(gas), len(nothing), len(brake)) if len(gas) != min(len(gas), len(nothing), len(brake)) else 0
    to_remove_nothing = 0#len(nothing) - min(len(gas), len(nothing), len(brake)) if len(nothing) != min(len(gas), len(nothing), len(brake)) else 0
    to_remove_brake = 0#len(brake) - min(len(gas), len(nothing), len(brake)) if len(brake) != min(len(gas), len(nothing), len(brake)) else 0
    del gas[:to_remove_gas]
    del nothing[:to_remove_nothing]
    del brake[:to_remove_brake]
    
    do_mods=False
    if do_mods:
        x=[0.44704, 1.78816, 3.57632]
        y=[.05, .1, .3]
        new_nothing = []
        for i in nothing:
            i[6]=.2
            new_nothing.append(i)
            '''if i[0] > i[2] and i[4] < 0.134112 and abs(i[0] - i[2]) > 0.89408: # if self car is faster than lead and we're not breaking and lead is not accelerating
                i[6] = np.interp(i[0]-i[2], x, y)
                new_nothing.append(i)
            else:
                new_nothing.append(i)'''
        nothing = list(new_nothing)
    driving_data = gas + nothing + brake
    


print(len(driving_data))
print()
#y_train = [i[5] - i[6] for i in driving_data]
y_train = [i[-3] - i[-2] for i in driving_data] # since some samples have a_rel, get gas and brake from end of list
print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
print("Brake samples: {}".format(len([i for i in y_train if i < 0])))
print("\nSamples from GM: {}, samples from other cars: {}".format(gm_counter, other_counter))

save_data = True
if save_data:
    print("Saving data...")
    save_dir="3model"
    x_train = [i[:5] for i in driving_data]
    #x_train = [i[:2] + [i[2] - i[0]] + i[-2:] for i in x_train] # makes index 2 be relative velocity
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(x_train, f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(y_train, f)
    os.remove(save_dir+"/normalized")
    print("Saved data!")

'''driving_data = [i for idx, i in enumerate(driving_data) if 20000 < idx < 29000]
x = [i for i in range(len(driving_data))]
y = [i[0] for i in driving_data]
plt.plot(x, y)
plt.show()'''