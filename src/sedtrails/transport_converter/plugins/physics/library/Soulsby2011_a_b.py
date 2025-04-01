"""
Functions to compute b (transition probability that a free particle becomes trapped [-])
and a (transition probability that a trapped particle becomes free [-]) using equations
from Soulsby et al. [2011].
"""

import numpy as np


def Soulsby2011_b(theta_max_Background,theta_cr_Background,theta_max_Tracer,soulsby2011_b_e,soulsby2011_theta_s):
    """
    Computes b (transition probability that a free particle becomes trapped [-]).
    """
    b = np.zeros(theta_max_Tracer.shape)
    for i in range(0,theta_max_Tracer.shape[0]):
        for j in range(0,theta_max_Tracer.shape[1]):
            if theta_max_Background[i][j] > theta_cr_Background:
                b[i][j] = soulsby2011_b_e * (1 - np.exp(\
                    -(theta_max_Background[i][j] - theta_cr_Background)\
                          / soulsby2011_theta_s))
            else:
                b[i][j] = 0
    
    return b


def Soulsby2011_a(b,soulsby2011_gamma_e):
    """
    Computes a (transition probability that a trapped particle becomes free [-]).
    """
    a = soulsby2011_gamma_e * b / (1 - soulsby2011_gamma_e)
    
    return a
