# b computation using Soulsby et al. [2011]

def Soulsby2011_b(theta_max_Background,theta_cr_Background,theta_max_Tracer):
    
    
    # import necessary packages
    import numpy as np
    
    
    # define empirical parameters for Soulsby calculations
    soulsby2011_b_e = 1.7e-7 # []
    soulsby2011_theta_s = 0.1 # []
    
    
    # compute b (transition probability that a free particle becomes trapped [-])
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
