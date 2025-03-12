# a computation using Soulsby et al. [2011]

def Soulsby2011_a(b):
    
    
    # import necessary packages
    import numpy as np
    
    
    # define empirical parameters for Soulsby calculations
    soulsby2011_gamma_e = 0.1 # []
    
    
    # compute a (transition probability that a trapped particle becomes free [-])
    a = soulsby2011_gamma_e * b / (1 - soulsby2011_gamma_e)
    
    return a
