import numpy as np
from numpy._typing._array_like import NDArray
import pytest
from sedtrails.transport_converter.plugins.physics.library.R_trial import Rfactor

# Case that we will be sure that R has to come out as zero
@pytest.fixture
def sample_data_case_1():
    """
    Defining some random initial values to do some tests
    """
    # Define a simple nondegenerate triangle.
    theta_max_A = np.array([[0.5, 0.7, 0.9, 0.1], 
    [0.4, 0.5, 0.3, 0.5],
    [0.3, 0.5, 0.1, 0.2]])
    theta_cr_A = float(1)
    w_s_t = float(0.05)
    u_star_m = np.array([[0.5, 0.5, 0.5, 0.5], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.5, 0.5, 0.5, 0.5]]) # I am not sure if it is a matrix or not
    u_star_max = np.array([[0.08, 0.08, 0.08, 0.08], 
    [0.08, 0.08, 0.08, 0.08],
    [0.08, 0.08, 0.08, 0.08]])
    Uc_mag = np.array([[4, 4, 4, 4], 
    [4, 4, 4, 4],
    [4, 4, 4, 4]])
    Ub = np.array([[0.5, 0.5, 0.5, 0.5], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.5, 0.5, 0.5, 0.5]]) # I am not sure if it is a matrix or not

    return [theta_max_A, theta_cr_A, w_s_t, u_star_m, u_star_max, Uc_mag, Ub]

def test_R_trial_1(sample_data_case_1: NDArray):
    """
    When theta_max_A < theta_cr_A, R should be zero
    """
    theta_max_A = sample_data_case_1[0]
    theta_cr_A = sample_data_case_1[1]
    w_s_t = sample_data_case_1[2]
    u_star_m = sample_data_case_1[3]
    u_star_max = sample_data_case_1[4]
    Uc_mag = sample_data_case_1[5]
    Ub = sample_data_case_1[6]

    def all_zeros(arr_list):
        return all(np.all(arr == 0) for arr in arr_list)

    assert all_zeros(Rfactor(
        theta_max_A, 
        theta_cr_A,
        w_s_t,
        u_star_m,
        u_star_max,
        Uc_mag,
        Ub
        ))

