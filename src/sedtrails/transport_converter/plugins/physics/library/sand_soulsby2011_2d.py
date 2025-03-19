# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:26:35 2025

@author: aguilera
"""

# Practice Soulsby, migrating from Matlab to Python

import numpy as np

# def sand_soulsby2011_2d(Ucx,Ucy,tau_m,tau_max,S):
    
#%% Grain properties
# Grain properties for tracer sediment

#%% R (Velocity Reduction Factor)

# I'll jump right into R since I don't have the structure of all the input variables now 
# and need some time to catch up

# some dummy matrices to get started with it
theta_max_A = np.array([[0.5, 0.7, 0.9, 0.1], 
    [0.4, 0.5, 0.3, 0.5],
    [0.3, 0.5, 0.1, 0.2]])

theta_cr_A = 0.2 

w_s_t = 0.1

u_star_m = np.array([[0.5, 0.5, 0.5, 0.5], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.5, 0.5, 0.5, 0.5]]) # I am not sure if it is a matrix or not

u_star_max = np.array([[0.08, 0.08, 0.08, 0.08], 
    [0.08, 0.08, 0.08, 0.08],
    [0.08, 0.08, 0.08, 0.08]])

Uc_mag = np.array([[4, 4, 4, 4], 
    [4, 4, 4, 4],
    [4, 4, 4, 4]])

# Soulsby rule of thumb: R_bedload ~ 0.15-0.50; R_suspload ~ 1.00

# Bedload velocity (Ub)
Ub = np.multiply(10*u_star_m,(1-0.7*np.sqrt(theta_cr_A/theta_max_A))) # [equation 7]

# Bedload velocity reduction factor (R_bedload)

Rb = np.zeros(Ub.shape)
Rs = np.zeros(Ub.shape)
R = np.zeros(Ub.shape)

for ii in range(0,theta_max_A.shape[0]):
    for jj in range(0,theta_max_A.shape[1]):
        if theta_max_A[ii][jj] > theta_cr_A: # [Equation 8]
            print(Ub[ii][jj]/Uc_mag[ii][jj])
            Rb[ii][jj] = Ub[ii][jj]/Uc_mag[ii][jj]
            if Rb[ii][jj] > 1:
                Rb[ii][jj] = 1 # apply velocity limiter
        else: 
            Rb[ii][jj] = 0

# Rouse parameter (B < 2.5 indicates complete suspension)
B = w_s_t/(0.4*u_star_max)

# Suspended load velocity reduction factor (R_suspload)
for ii in range(0,theta_max_A.shape[0]):
    for jj in range(0,theta_max_A.shape[1]):
        if Rb[ii][jj] == 0:
            Rs[ii][jj] = 0
        else:
            # didn't find a good way of presenting this huge equation readable in python!!!!
            Rs[ii][jj] = np.multiply(np.divide(np.multiply(Rb[ii][jj],(1-B[ii][jj])),
                                               (8/7-B[ii][jj])),
                                     np.divide((np.power((8/7*Rb[ii][jj]),(8-7*B[ii][jj]))-1),
                                              (np.power((8/7*Rb[ii][jj]),(7-7*B[ii][jj]))-1)))
        
        # Apply velocity limiter (grain velocity cannot exceed flow velocity)
        if Rs[ii][jj] > 1:
            Rs[ii][jj] = 1
        elif np.isnan(Rs[ii][jj]): # I am not sure about this computation
            Rs[ii][jj] = 0
            
# Reduction factor R
for ii in range(0,theta_max_A.shape[0]):
    for jj in range(0,theta_max_A.shape[1]):
        if B[ii][jj] < 2.5: # if all material is in suspension use Rs (for suspended load)
            R[ii][jj] = Rs[ii][jj]
        else: # otherwise use Rb (for bed load)
            R[ii][jj] = Rb[ii][jj]
# print(Rb)
# print(Rs)
# print(R)
# print(B)