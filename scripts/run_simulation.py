import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.quantum_states import *
from src.theory.discriminator import *


# region parameter configuration
with open("../config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

opt_method = config['optimization']['method']
tol = config['optimization']['tol']
rhobeg = eval(config['optimization']['rhobeg'])
max_iter = int(float(config['optimization']['maxiter']))

lambda_val = config['simulation']['lambda_val']

columns = config['columns']
# endregion

# region toy example
s_parameter = 0.75
prior_probability = [0.5, 0.5, 0]
prepared_state_set = np.array([
    [np.sqrt((1+s_parameter)/2), np.sqrt((1-s_parameter)/2), 0],
    [np.sqrt((1+s_parameter)/2), -np.sqrt((1-s_parameter)/2), 0]
])
overlap = s_parameter
rho_list = get_rho_list(prepared_state_set)
dim = len(prior_probability)
# endregion


# region theory curve
theory_df = pd.DataFrame(columns=columns['theory'])
for fixed_rate in np.linspace(0, 1, 100):
    optimal_measurements = solve_sdp_bound(prepared_state_set, prior_probability, dim, fixed_rate)

    P_success, P_error, P_fail = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)
    row = [overlap, fixed_rate, P_success, P_error, P_fail]

    theory_df = pd.concat([theory_df, pd.DataFrame([row], columns=columns['theory'])], ignore_index=True)
# endregion


for fixed_rate in np.linspace(0, overlap, 50):
    obj = cobyla_objective

    initial_parameter = np.random.uniform(0, 2*np.pi, size=((dim**2) - 1))

    result = minimize(
        obj,
        initial_parameter,
        args=(prepared_state_set, prior_probability, dim, fixed_rate, lambda_val),
        method=opt_method,
        tol=tol,
        options={
            'rhobeg': rhobeg,
            'maxiter': max_iter,
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
