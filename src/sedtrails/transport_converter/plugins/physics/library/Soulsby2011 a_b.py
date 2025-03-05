# Soulsby et al. [2011] sand grain velocity calculation (a and b part)

import numpy as np

#def sand_soulsby2011_2d(Ucx,Ucy,tau_m,tau_max,S):

# Grain properties for tracer sediment
#Dstar_t = (S.g * (S.rhoS / S.rhoW - 1) / (S.visc_kin ** 2)) ** (1/3) * S.dTracer
#theta_cr_t = 0.3 / (1 + 1.2 * Dstar_t) + 0.055 * (1 - np.exp(-0.020 * Dstar_t))
#w_s_t = (S.visc_kin / S.dTracer) * (np.sqrt(10.36 ** 2 + 1.049 * (Dstar_t ** 3)) - 10.36)
#A = S.dTracer / S.dBackground
#theta_cr_A = theta_cr_t * np.sqrt(8 / (3 * (A ** 2) + 6 * A - 1)) * ((3.2260 * A) / (4 * A - 2 * (A + 1 - np.sqrt(A ** 2 + 2 * A - 1/3))))

# Grain properties for background sediment
#Dstar_a = (S.g * (S.rhoS / S.rhoW - 1) / (S.visc_kin ** 2)) ** (1/3) * S.dBackground
#theta_cr_a = 0.3 / (1 + 1.2 * Dstar_a) + 0.055 * (1 - np.exp(-0.020 * Dstar_a))
#w_s_a = (S.visc_kin / S.dBackground) * (np.sqrt(10.36 ** 2 + 1.049 * (Dstar_a ** 3)) - 10.36)

# Velocity magnitudes
#Uc_mag = np.sqrt(Ucx ** 2 + Ucy ** 2)

# Shear stresses
#tau_m = np.abs(tau_m)
#tau_max = np.abs(tau_max)

# Friction velocities
#u_star_m = np.sqrt(tau_m / S.rhoW)
#u_star_max = np.sqrt(tau_max / S.rhoW)

#Shield parameters
#theta_m_A = tau_m / (S.g * (S.rhoS - S.rhoW) * S.dTracer)
#theta_max_A = tau_max / (S.g * (S.rhoS - S.rhoW) * S.dTracer)
#theta_m_a = tau_m / (S.g * (S.rhoS - S.rhoW) * S.dBackground)
#theta_max_a = tau_max / (S.g * (S.rhoS - S.rhoW) * S.dBackground)

#stole dummy matrices from Monica's code to replace unavailable values above
theta_max_A = np.array([[0.5, 0.7, 0.9, 0.1], 
    [0.4, 0.5, 0.3, 0.5],
    [0.3, 0.5, 0.1, 0.2]])

theta_cr_A = 0.2 

theta_max_a = np.array([[0.5, 0.7, 0.9, 0.1], 
    [0.4, 0.5, 0.3, 0.5],
    [0.3, 0.5, 0.1, 0.2]])

theta_cr_a = 0.2

soulsby2011_b_e = 1.7e-7
soulsby2011_theta_s = 0.1
soulsby2011_gamma_e = 0.1

#if S.soulsby2011_bedInteraction: This is commented because I don't know where this will be defined in the new code
    
b = np.zeros(theta_max_A.shape) # b = transition probability that a free particle becomes trapped [-]
for ii in range(0,theta_max_A.shape[0]):
    for jj in range(0,theta_max_A.shape[1]):
        if theta_max_a[ii][jj] > theta_cr_a:
            b[ii][jj] = soulsby2011_b_e * (1 - np.exp(-(theta_max_a[ii][jj] - theta_cr_a) / soulsby2011_theta_s)) #should be S.Soulsby2011_b_e and S.soulsby2011_theta_s
        else:
            b[ii][jj] = 0
    
    a = soulsby2011_gamma_e * b / (1 - soulsby2011_gamma_e) # a = transition probability that a trapped particle becomes free [-], should be S.soulsby2011_gamma_e

#else:
#    b = np.nan(theta_max_A.shape)
#    a = np.nan(theta_max_A.shape)
