import numpy as np
scale = [0.0, 1.0] # wider scale might improve accuracy
def normX(data, data_scale=None):
    if data_scale==None:
        v = [inner for outer in data for inner in [outer[0]]+[outer[2]]] # all vel data
        a = [inner for outer in data for inner in [outer[1]]+[outer[4]]] # all accel data
        x = [i[3] for i in data] # all distance data
        
        v_scale = [min(v), max(v)]
        a_scale = [min(a), max(a)]
        x_scale = [min(x), max(x)]
        
        normalized = [[np.interp(i[0], v_scale, scale), np.interp(i[1], a_scale, scale), np.interp(i[2], v_scale, scale), np.interp(i[3], x_scale, scale), np.interp(i[4], a_scale, scale)] for i in data]
        return {'v_scale': v_scale, 'a_scale': a_scale, 'x_scale': x_scale, 'normalized': normalized}
    else:
        y = [data_scale[0], data_scale[1]]
        return np.interp(data, y, scale)


#print(normX([[9.387969017029, 1.137865662575, 12.12175655365, 30.25, 3.817160964012], [18.121639251709, -0.024567155167, 19.496248245239, 14.0, 0.072589494288]]))