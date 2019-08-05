import itertools
#import time
import numpy as np
scale = [0.0, 1.0] # wider scale might improve accuracy

def interp(x, xp, fp):
  N = len(xp)
  def get_interp(xv):
    hi = 0
    while hi < N and xv > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi == N and xv > xp[low] else (
      fp[0] if hi == 0 else 
      (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])
  return [get_interp(v) for v in x] if hasattr(
    x, '__iter__') else get_interp(x)

def norm(data, data_scale=None):
    if data_scale==None:
        v_ego = [inner[0] for outer in data for inner in outer]
        v_lead = [inner[1] for outer in data for inner in outer]
        #all_a = list(itertools.chain.from_iterable(([[inner[1], inner[4]] for outer in data for inner in outer])))
        #all_a = list(itertools.chain.from_iterable(([[inner[4]] for outer in data for inner in outer])))
        x_lead = [inner[2] for outer in data for inner in outer]
        #for i in data:
            #for x in i:
                #all_v.append(x[0])
                #all_v.append(x[2])
                #all_a.append(x[1])
                #all_a.append(x[4])
                #all_x.append(x[3])
        scales = {     
                'v_ego_scale': [min(v_ego), max(v_ego)],
                'v_lead_scale': [min(v_lead), max(v_lead)],
                'x_lead_scale': [min(x_lead), max(x_lead)]
                }
        
        #normalized = [[[interp(d[0], v_scale, scale), interp(d[1], a_scale, scale), interp(d[2], v_scale, scale), interp(d[3], x_scale, scale), interp(d[4], a_scale, scale)] for d in i] for i in data]
        normalized = [[[interp(d[0], scales['v_ego_scale'], scale), interp(d[1], scales['v_lead_scale'], scale), interp(d[2], scales['x_lead_scale'], scale)] for d in i] for i in data]
        return {'scales': scales, 'normalized': np.array(normalized)}
    else:
        y = [data_scale[0], data_scale[1]]
        return interp(data, y, scale)

'''a=norm([[[0.073444232345, 0.294612795115, 1.125],
  [0.154977455735, 0.51556789875, 1.125]],
 [[0.154977455735, 0.51556789875, 1.125],
  [0.232993260026, 0.599586844444, 2.25]]])
print(a)'''