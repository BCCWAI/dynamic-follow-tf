#import itertools
#import time
#import numpy as np
#scale = [0.0, 1.0] # wider scale might improve accuracy

def norm(x):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    return x

'''def interp(x, xp, fp):
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
    x, '__iter__') else get_interp(x)'''

'''def norm(data, data_scale=None):
    if data_scale==None:
        all_v = list(itertools.chain.from_iterable(([[inner[0], inner[2]] for outer in data for inner in outer])))
        all_a = list(itertools.chain.from_iterable(([[inner[1], inner[4]] for outer in data for inner in outer])))
        all_x = [inner[3] for outer in data for inner in outer]
        #for i in data:
            #for x in i:
                #all_v.append(x[0])
                #all_v.append(x[2])
                #all_a.append(x[1])
                #all_a.append(x[4])
                #all_x.append(x[3])
        
        v_scale = [min(all_v), max(all_v)]
        a_scale = [min(all_a), max(all_a)]
        x_scale = [min(all_x), max(all_x)]
        
        normalized = [[[interp(d[0], v_scale, scale), interp(d[1], a_scale, scale), interp(d[2], v_scale, scale), interp(d[3], x_scale, scale), interp(d[4], a_scale, scale)] for d in i] for i in data]
        return {'v_scale': v_scale, 'a_scale': a_scale, 'x_scale': x_scale, 'normalized': normalized}
    else:
        y = [data_scale[0], data_scale[1]]
        return interp(data, y, scale)'''
