# sand grain velocity calculation using Soulsby et al. [2011]


# import necessary packages
import numpy as np


#import necessary functions
import Soulsby2011_b
import Soulsby2011_a


# create dummy matrices for calculations (these variables should be pre-computed somewhere)

# maximum values of Shield parameter for tracer sediment [-]
theta_max_Tracer = np.array([[0.5, 0.7, 0.9, 0.1],
    [0.4, 0.5, 0.3, 0.5],
    [0.3, 0.5, 0.1, 0.2]])

# critical value of Shield parameter for tracer sediment [-]
theta_cr_Tracer = 0.2

#  maximum values of Shield parameter for background sediment [-]
theta_max_Background = np.array([[0.5, 0.7, 0.9, 0.1],
    [0.4, 0.5, 0.3, 0.5],
    [0.3, 0.5, 0.1, 0.2]])

# critical value of Shield parameter for background sediment [-]
theta_cr_Background = 0.2


# pre-compute a and b
b = Soulsby2011_b
a = Soulsby2011_a
