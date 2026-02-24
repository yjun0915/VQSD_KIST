import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.hardware import slm_core
from src.utils.quantum_states import *
from src.theory.discriminator import *
from src.hardware.tcspc_core import *
from src.hardware.slm_core import *


trial_num = 10
s_parameter = 0.75
prior_probability = [0.5, 0.5, 0]
prepared_state_set = np.array([
    [np.sqrt((1+s_parameter)/2), np.sqrt((1-s_parameter)/2), 0],
    [np.sqrt((1+s_parameter)/2), -np.sqrt((1-s_parameter)/2), 0]
])
overlap = s_parameter
rho_list = get_rho_list(prepared_state_set)
dim = len(prior_probability)

results = pd.DataFrame(index=range(1, trial_num+1), columns=["success rate", "error rate", "failure rate"])

with timetagger_session(500, 50, 2, [0, 0]) as timetagger:
    with slm_session() as slm:
        experiment = Experiment(timetagger, slm, prepared_state_set, dim)
        for res_idx in range(1, trial_num+1):
            for fixed_rate in np.linspace(0, overlap, 5):
                obj = experiment.cobyla_objective

                initial_parameter = np.random.uniform(0, 2*np.pi, size=((dim**2) - 1))

                result = minimize(
                    obj,
                    initial_parameter,
                    args=(prior_probability, fixed_rate, 3),
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

                results.loc[res_idx] = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)

from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_name = f"experiment_results_{timestamp}.csv"
results.to_csv(file_name)
