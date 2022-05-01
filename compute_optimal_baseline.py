# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:30:39 2021

@author: USER
"""

import numpy as np

#(a),(b)
def compute_trace(baseline):
    pi = [0.1,0.5,0.4]
    r = [100-baseline,98-baseline,95-baseline]

    sum = np.zeros((3,3))
    mean =  np.zeros((3,1))

    for a in range(3):
        V = [0,0,0]
        for row in range(3):
            if a==row:
                V[row] = 1-pi[row]
            else :
                V[row] = -pi[row]
    
        V = np.array(V).reshape((3,1))
    
        sum = sum + pi[a]*(r[a])**2*np.dot(V,V.T)
        mean = mean + pi[a]*r[a] * V

    #print(sum - np.dot(mean,mean.T))

    trace = (sum - np.dot(mean,mean.T))[0][0]+(sum - np.dot(mean,mean.T))[1][1]+(sum -   np.dot(mean,mean.T))[2][2]

    return trace



#(c)
def iteration(middle,bound):

    if bound < 10**-9:
        return compute_trace(middle)
    
    upper = middle + bound
    botton = middle - bound
    print(" 上: ",upper," 下: ",botton," 中: ",middle," Bound: ",bound)
    up = compute_trace(upper)
    print("upper: ",up)
    bot = compute_trace(botton)
    print("botton: ",bot)
    mid = compute_trace(middle)
    print("middle: ",mid)
    bound = bound/2
    
    if up < mid :
        return iteration(upper,bound)
    elif bot < mid :
        return iteration(botton,bound)
    else:
        return iteration(middle,bound)
    

min  = iteration(97,1)
print(min)

