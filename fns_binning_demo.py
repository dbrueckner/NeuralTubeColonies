#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:12:43 2022

@author: D.Brueckner
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def calc_center(x,y,field):
    N_x,N_y = field.shape
    total_intensity = field.sum()
    center = np.zeros(2)
    for i in range(0,N_x):
        for j in range(0,N_y):
            center[0] += field[i,j]*x[i]
            center[1] += field[i,j]*y[j]
    center = center/total_intensity
    return center

@jit(nopython=True)
def calc_profile_pizza(x,y,field,center,N_bins_r,N_bins_phi):
    N_x,N_y = field.shape
    x_max = max(x)
    bins_r = np.linspace(0,x_max,N_bins_r)
    delta_r = (bins_r[1]-bins_r[0])/2
    
    width = 2*np.pi/N_bins_phi
    bins_phi = np.linspace(0,2*np.pi-width,N_bins_phi)
    delta_phi = (bins_phi[1]-bins_phi[0])/2
    
    sum_r = np.zeros((N_bins_r,N_bins_phi))
    N_r = np.zeros((N_bins_r,N_bins_phi))
    
    intensity_radius = np.zeros((N_bins_r,N_bins_phi))
    
    for i in range(0,N_x):
        for j in range(0,N_y):
            radius = np.sqrt((x[i]-center[0])**2 + (y[j]-center[1])**2)
            
            #phi has 0 on the left x-axis and goes from 0 to 2pi
            phi = np.arctan2( (y[j]-center[1]),(x[i]-center[0]) ) + np.pi
            if radius < x_max:
                
                for b_r in range(0,N_bins_r):
                    if(radius > bins_r[b_r]-delta_r and radius < bins_r[b_r]+delta_r):
                        if b_r == 0: #don't divide slices for innermost ring
                            sum_r[b_r,:] += field[i,j]
                            N_r[b_r,:] += 1
                        else:
                            for b_phi in range(0,N_bins_phi):
                                if(phi > bins_phi[b_phi]-delta_phi and phi < bins_phi[b_phi]+delta_phi):
                                    sum_r[b_r,b_phi] += field[i,j]
                                    N_r[b_r,b_phi] += 1

    
    for b_r in range(0,N_bins_r):
        for b_phi in range(0,N_bins_phi):
            if N_r[b_r,b_phi] > 0:
                intensity_radius[b_r,b_phi] = sum_r[b_r,b_phi]/N_r[b_r,b_phi]
    return bins_r,bins_phi,intensity_radius
