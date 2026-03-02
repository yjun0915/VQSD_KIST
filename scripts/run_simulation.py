import time
import yaml
import pandas as pd

from datetime import datetime
from pathlib import Path
from tqdm import trange
from scipy.optimize import minimize

from src.utils.quantum_states import *
from src.utils.messenger import *
from src.theory.discriminator import *


# region parameter configuration
with open("../config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

opt_config = config['optimization']['COBYLA']

lambda_val = config['minimize']['lambda_val']

minimize_params = config['minimize']

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


# region theory data
theory_df = pd.DataFrame(columns=columns['theory'])
theory_dir = Path(f"../data/theory/dim_{dim}")
theory_dir.mkdir(parents=True, exist_ok=True)
theory_filename = f"ov{overlap}.csv"
theory_filepath = theory_dir / theory_filename
for fixed_rate in np.linspace(0, 1, 100):
    optimal_measurements = solve_sdp_bound(prepared_state_set, prior_probability, dim, fixed_rate)
    P_success, P_error, P_fail = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)
    new_row = [overlap, fixed_rate, P_success, P_error, P_fail]
    theory_df = pd.concat([theory_df, pd.DataFrame([new_row], columns=columns['theory'])], ignore_index=True)
theory_df.to_csv(theory_filepath, index=False)
# endregion


start = time.time()
# region simulation data
sim_df = pd.DataFrame(columns=columns['sim data'])
sim_history_df = pd.DataFrame([])

current_time = datetime.now().strftime("%y%m%d_%H%M")
sim_dir = Path(f"../data/simulation/dim_{dim}")
sim_dir.mkdir(parents=True, exist_ok=True)

sim_filename = f"{current_time}_ov{overlap}.csv"
sim_history_filename = f"{current_time}_ov{overlap}_history.csv"

sim_filepath = sim_dir / sim_filename
sim_history_filepath = sim_dir / sim_history_filename

q_points = minimize_params['q_points']
fixed_rates = np.linspace(0, overlap, q_points)
best_lagrangians = np.full(q_points, -np.inf)
best_histories = [[] for _ in range(q_points)]
for trial in trange(minimize_params['trial'], desc="Trials"):
    for fr_idx, fixed_rate in enumerate(fixed_rates):
        parameter_history = []


        def tracking_objective(x, *args):
            current_lagrangian = cobyla_objective(x, *args)
            parameter_history.append(x.copy().tolist() + [-current_lagrangian])
            return current_lagrangian

        initial_parameter = np.random.uniform(0, 2*np.pi, size=((dim**2) - 1))
        result = minimize(
            fun=tracking_objective,
            x0=initial_parameter,
            args=(prepared_state_set, prior_probability, dim, fixed_rate, lambda_val),
            **opt_config
        )
        lagrangian = -result.fun

        vector_list = unitary_matrix(result.x, dim).T
        optimal_measurements = {}
        for vector_idx, vector in enumerate(vector_list):
            optimal_measurements[f"M{np.mod(vector_idx + 1, dim)}"] = vector

        P_success, P_error, P_fail = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)
        new_row = [overlap, trial, fixed_rate, P_success, P_error, P_fail, lagrangian, lambda_val, opt_config['method'], result.nfev]
        sim_df = pd.concat([sim_df, pd.DataFrame([new_row], columns=columns['sim data'])], ignore_index=True)

        if lagrangian > best_lagrangians[fr_idx]:
            best_histories[fr_idx] = parameter_history
            best_lagrangians[fr_idx] = lagrangian
history_dict = {f"{fixed_rates[i]:.3f}": pd.Series(best_histories[i]) for i in range(q_points)}
sim_history_df = pd.DataFrame(history_dict)
sim_history_df.columns = pd.MultiIndex.from_product([["fixed rate"], sim_history_df.columns])

sim_df.to_csv(sim_filepath, index=False)
sim_history_df.to_csv(sim_history_filepath, index=False)
# endregion

best_idx = sim_df['lagrangian'].idxmax()
best_fixed_rate = sim_df.loc[best_idx, 'fixed rate']
max_P_succ = sim_df.loc[best_idx, 'success rate']
theory_P_succ = theory_df.iloc[(theory_df['fixed rate'] - best_fixed_rate).abs().argsort()[:1]]['success rate'].values[0]
avg_lag = sim_df['lagrangian'].mean()
elapsed_time_raw = time.time() - start
minutes, seconds = divmod(int(elapsed_time_raw), 60)
time_str = f"{minutes}m {seconds}s"
send_message(dim, overlap, opt_config['method'], minimize_params['lambda_val'], max_P_succ, theory_P_succ, avg_lag, time_str, minimize_params['trial'], sim_filename)
