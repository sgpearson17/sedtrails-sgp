# For now I will start this as a function and later on I will shift it to a class
# if I see it is possible

import numpy as np
from numpy.typing import NDArray


def Rfactor(
    theta_max_A: NDArray[np.float64],
    theta_cr_A: float,
    w_s_t: float,
    u_star_m: NDArray[np.float64],
    u_star_max: NDArray[np.float64],
    Uc_mag: NDArray[np.float64],
    Ub: NDArray[np.float64],
) -> NDArray:
    
    """
    Computes a reduction factor which is applied on the flow velocities to obtain grain velocities

    Returns
    -------
    R (NDArray): Reduction factor
    """
    Rb = np.zeros(Ub.shape)
    Rs = np.zeros(Ub.shape)
    R = np.zeros(Ub.shape)

    for ii in range(0, theta_max_A.shape[0]):
        for jj in range(0, theta_max_A.shape[1]):
            if theta_max_A[ii][jj] > theta_cr_A:  # [Equation 8]
                print(Ub[ii][jj] / Uc_mag[ii][jj])
                Rb[ii][jj] = Ub[ii][jj] / Uc_mag[ii][jj]
                if Rb[ii][jj] > 1:
                    Rb[ii][jj] = 1  # apply velocity limiter
            else:
                Rb[ii][jj] = 0

    # Rouse parameter (B < 2.5 indicates complete suspension)
    B = w_s_t / (0.4 * u_star_max) # I think this can also be computed somewhere else

    # Suspended load velocity reduction factor (R_suspload)
    for ii in range(0, theta_max_A.shape[0]):
        for jj in range(0, theta_max_A.shape[1]):
            if Rb[ii][jj] == 0:
                Rs[ii][jj] = 0
            else:
                # didn't find a good way of presenting this huge equation readable in python!!!!
                Rs[ii][jj] = np.multiply(
                    np.divide(
                        np.multiply(Rb[ii][jj], (1 - B[ii][jj])), (8 / 7 - B[ii][jj])
                    ),
                    np.divide(
                        (np.power((8 / 7 * Rb[ii][jj]), (8 - 7 * B[ii][jj])) - 1),
                        (np.power((8 / 7 * Rb[ii][jj]), (7 - 7 * B[ii][jj])) - 1),
                    ),
                )

            # Apply velocity limiter (grain velocity cannot exceed flow velocity)
            if Rs[ii][jj] > 1:
                Rs[ii][jj] = 1
            elif np.isnan(Rs[ii][jj]):  # I am not sure about this computation
                Rs[ii][jj] = 0

    # Reduction factor R
    for ii in range(0, theta_max_A.shape[0]):
        for jj in range(0, theta_max_A.shape[1]):
            if (
                B[ii][jj] < 2.5
            ):  # if all material is in suspension use Rs (for suspended load)
                R[ii][jj] = Rs[ii][jj]
            else:  # otherwise use Rb (for bed load)
                R[ii][jj] = Rb[ii][jj]
    return Rb, Rs, R 
