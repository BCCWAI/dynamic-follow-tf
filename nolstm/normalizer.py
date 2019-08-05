import numpy as np
scale = [0.0, 1.0] # wider scale might improve accuracy
def normX(data, data_scale=None):
    if data_scale==None:
        v_ego = [i[0] for i in data]
        #a_ego = [i[1] for i in data]
        v_lead = [i[2] for i in data]
        x_lead = [i[3] for i in data]
        a_lead = [i[4] for i in data]
        #a_rel = [i[5] for i in data]
        
        v_ego_scale = [min(v_ego), max(v_ego)]
        #a_ego_scale = [min(a_ego), max(a_ego)]
        v_lead_scale = [min(v_lead), max(v_lead)]
        x_lead_scale = [min(x_lead), max(x_lead)]
        a_lead_scale = [min(a_lead), max(a_lead)]
        #a_rel_scale = [min(a_rel), max(a_rel)]
        
        #all: normalized = [[np.interp(i[0], v_ego_scale, scale), np.interp(i[1], a_ego_scale, scale), np.interp(i[2], v_lead_scale, scale), np.interp(i[3], x_lead_scale, scale), np.interp(i[4], a_lead_scale, scale), np.interp(i[5], a_rel_scale, scale)] for i in data]
        normalized = [[np.interp(i[0], v_ego_scale, scale), np.interp(i[2], v_lead_scale, scale), np.interp(i[3], x_lead_scale, scale), np.interp(i[4], a_lead_scale, scale)] for i in data]
        scales = {'v_ego_scale': v_ego_scale,
                #'a_ego_scale': a_ego_scale,
                'v_lead_scale': v_lead_scale,
                'x_lead_scale': x_lead_scale,
                'a_lead_scale': a_lead_scale
                #'a_rel_scale': a_rel_scale
                }
        return {'scales': scales, 'normalized': np.array(normalized)}
    else:
        y = [data_scale[0], data_scale[1]]
        return np.interp(data, y, scale)

def normXOld(data, data_scale=None):
    if data_scale==None:
        v = [inner for outer in data for inner in [outer[0]]+[outer[2]]] # all vel data
        #a = [inner for outer in data for inner in [outer[1]]+[outer[4]]] # all accel data
        a = [i[4] for i in data] # all lead accel data
        x = [i[3] for i in data] # all distance data
        
        v_scale = [min(v), max(v)]
        a_scale = [min(a), max(a)]
        x_scale = [min(x), max(x)]
        
        normalized = [[np.interp(i[0], v_scale, scale), np.interp(i[1], a_scale, scale), np.interp(i[2], v_scale, scale), np.interp(i[3], x_scale, scale), np.interp(i[4], a_scale, scale)] for i in data]
        #normalized = [[np.interp(i[0], v_scale, scale), np.interp(i[2], v_scale, scale), np.interp(i[3], x_scale, scale), np.interp(i[4], a_scale, scale)] for i in data]
        return {'v_scale': v_scale, 'a_scale': a_scale, 'x_scale': x_scale, 'normalized': np.array(normalized)}
    else:
        y = [data_scale[0], data_scale[1]]
        return np.interp(data, y, scale)


#a = normX([[9.387969017029, 1.137865662575, 12.12175655365, 30.25, 3.817160964012, 1], [18.121639251709, -0.024567155167, 19.496248245239, 14.0, 0.072589494288, 0]])
#print(a)