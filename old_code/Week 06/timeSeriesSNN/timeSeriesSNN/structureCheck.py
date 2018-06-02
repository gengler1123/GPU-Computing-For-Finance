# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:01:41 2017

@author: Gary
"""

import pandas as pd
import matplotlib.pyplot as plt
import subprocess




numNeurons = 1000
runTime = 1000
maxDelay = 15
numThreads = 512
transientTime = 200
PATH = "Results.csv"
prop_excit = 0.8

p_e2e = 0.6
p_e2i = 0.6
p_i2e = 0.8
p_i2i = 0.5


w_e_max = 200
w_i_max = 300
tau_e = 1.5
tau_i = 1
x_max = 3.0
y_max = 3.0
dt = 0.25



args = ["../Debug/timeSeriesSNN.exe",
        str(numNeurons), 
        str(runTime), 
        str(maxDelay), 
        str(numThreads), 
        str(transientTime),
        str(PATH), 
        str(prop_excit),
        str(p_e2e), 
        str(p_e2i), 
        str(p_i2e),
        str(p_i2i),
        str(w_e_max), 
        str(w_i_max), 
        str(tau_e), 
        str(tau_i), 
        str(x_max), 
        str(y_max),
        str(dt)]

subprocess.call(args)


#net = pd.read_csv("netStructure.csv")
Fir = pd.read_csv(PATH, header=None)
#neu = pd.read_csv("neurons.csv")



plt.figure()
plt.scatter(Fir[0],Fir[1])
plt.show()

#plt.figure()
#plt.scatter(neu['x'],neu['y'])
#plt.show() 