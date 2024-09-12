#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:16:36 2023

@author: D.Brueckner

"""

import numpy as np
import fns_sim as fns_sim
model = fns_sim.func_sim
  
rad = 1
R_max = 5*rad
t_max = 96

actLbyS = 1
actBbyS = 1

actBbyL = 0.05
actLbyW = 5

KhillL = 30
KHillN = 0.3
KhillS = 0.001

HillL = 2
HillS = 2

D_b = 0.001
D_n = 0.001
D_w = 0.001

b0 = 0.25

t_b = 5
t_n = 10
t_s = 0.63
t_l = 6.7
t_w = 8.1

degB = 1/t_b
degN = 1/t_n
degS = 1/t_s
degL = 1/t_l
degW = 1/t_w

A = 0
sigma = 0.1
delta_t = 0.1
N_t = int(t_max/delta_t)
N_cells = 200
oversampling = 500
dx = R_max/N_cells

params = (D_b,D_n,D_w,actBbyS,degB,degN,degS,degW,actBbyL,degL,b0,A,sigma,rad,actLbyS,KhillS,HillS,KhillL,HillL,actLbyW,KHillN)
modes = (N_cells,N_t,delta_t,oversampling,dx)

import matplotlib.pyplot as plt
plt.close('all')
fs = 12
fs2 = 12   
params_fig = {
          'font.size':   fs,
          'xtick.labelsize': fs2,
          'ytick.labelsize': fs2,
          }
plt.rcParams.update(params_fig)
file_suffix = '.pdf'

colors = ['royalblue','orange','red','limegreen','grey']

time = np.arange(N_t)*delta_t
xx = np.arange(N_cells)*dx

def activation(x,K,h): return x**h/(K**h+x**h)

mode_perturb = 'WT'
X = model(params,modes,mode_perturb) #

fig_size = [4,3]
params_fig = {'figure.figsize': fig_size,}
plt.rcParams.update(params_fig)

plt.figure()
for m in [0,1,2,3,4]:
    trajectory = np.max(X[:,:,m],axis=0)
    max_amplitude = np.max(trajectory)
    plt.plot(time,trajectory/max_amplitude,color=colors[m],lw=2)

plt.xlabel(r'time (h)')
plt.ylabel(r'concentration $c/c_\mathrm{max}$')

plt.tight_layout()


    
