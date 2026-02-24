import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.utils.quantum_states import *
from src.theory.discriminator import *
from src.hardware.tcspc_core import *


s_parameter = 0.75
prior_probability = [0.5, 0.5, 0]
prepared_state_set = np.array([
    [np.sqrt((1+s_parameter)/2), np.sqrt((1-s_parameter)/2), 0],
    [np.sqrt((1+s_parameter)/2), -np.sqrt((1-s_parameter)/2), 0]
])
overlap = s_parameter
rho_list = get_rho_list(prepared_state_set)
dim = len(prior_probability)

plt.figure(1)


# region theory curve
for fixed_rate in np.linspace(0, 1, 100):
    optimal_measurements = solve_sdp_bound(prepared_state_set, prior_probability, dim, fixed_rate)

    P_success, P_error, P_fail = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)

    plt.scatter(fixed_rate, P_success, color='dodgerblue')
    plt.scatter(fixed_rate, P_error, color='firebrick')
    plt.scatter(fixed_rate, P_fail, color='limegreen')
# endregion

for fixed_rate in np.linspace(0, overlap, 50):
    obj = cobyla_objective

    initial_parameter = np.random.uniform(0, 2*np.pi, size=((dim**2) - 1))

    result = minimize(
        obj,
        initial_parameter,
        args=(prepared_state_set, prior_probability, dim, fixed_rate, 3),
        method='COBYLA',
        tol=0.01,
        options={
            'rhobeg': np.pi/2,
            'maxiter': 1e3,
            'disp': False,
        }
    )

    vector_list = unitary_matrix(result.x, dim).T

    optimal_measurements = {}
    for vector_idx, vector in enumerate(vector_list):
        optimal_measurements[f"M{np.mod(vector_idx + 1, dim)}"] = vector

    P_success, P_error, P_fail = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)

    plt.scatter(fixed_rate, P_success, color='k', s=10)
    plt.scatter(fixed_rate, P_error, color='r', s=10)
    plt.scatter(fixed_rate, P_fail, color='k', s=10)

plt.show()
