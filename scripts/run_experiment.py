import time
import yaml

import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from tqdm import trange
from scipy.optimize import minimize

from src.utils.quantum_states import *
from src.utils.messenger import *
from src.theory.discriminator import *
from src.hardware.tcspc_core import *
from src.hardware.slm_core import *


cw = 500

# region parameter configuration
with open("../config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

opt_config = config['optimization']['COBYLA']

lambda_val = config['minimize']['lambda_val']

minimize_params = config['minimize']

columns = config['columns']
# endregion


activate = {
    'toy_example_1': True,
    'toy_example_2': False,
    'toy_example_3': False
}

# region toy example 1
if activate['toy_example_1']:
    s_parameter = 0.75
    prior_probability = [0.5, 0.5, 0]
    prepared_state_set = np.array([
        [np.sqrt((1+s_parameter)/2), np.sqrt((1-s_parameter)/2), 0],
        [np.sqrt((1+s_parameter)/2), -np.sqrt((1-s_parameter)/2), 0]
    ])
    overlap = s_parameter
# endregion

# region toy example 2
elif activate['toy_example_2']:
    s_parameter = 0.75
    prior_probability = [0.5, 0.5, 0, 0]
    prepared_state_set = np.array([
        [np.sqrt((1+s_parameter)/2), np.sqrt((1-s_parameter)/2), 0, 0],
        [np.sqrt((1+s_parameter)/2), -np.sqrt((1-s_parameter)/2), 0, 0],
        [0, 0, 0, 0]
    ])
    overlap = s_parameter
# endregion

# region toy example 3
elif activate['toy_example_3']:
    theta = 2 * np.pi / 3
    varphi = 2 * np.pi / 3
    xi = np.pi * 0.365

    prior_probability = [1/3, 1/3, 1/3, 0]
    prepared_state_set = np.array([
        [np.cos(xi), 0, np.sin(xi), 0],
        [np.cos(xi) * np.cos(theta), np.cos(xi) * np.sin(theta), np.sin(xi), 0],
        [np.cos(xi) * np.cos(varphi), -np.cos(xi) * np.sin(varphi), np.sin(xi), 0]
    ])
    overlap = -0.5*(np.cos(xi)**2) + (np.sin(xi)**2)
# endregion

else:
    print("no example was selected")
    quit()

rho_list = get_rho_list(prepared_state_set)
dim = len(prior_probability)

# region theory data
theory_df = pd.DataFrame(columns=columns['theory'])
theory_dir = Path(f"../data/theory/dim_{dim}")
theory_dir.mkdir(parents=True, exist_ok=True)
theory_filename = f"ov{overlap:.2f}.csv"
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
raw_df = pd.DataFrame(columns=columns['raw data'])
raw_history_df = pd.DataFrame([])

current_time = datetime.now().strftime("%y%m%d_%H%M")
raw_dir = Path(f"../data/experiment/dim_{dim}")
raw_dir.mkdir(parents=True, exist_ok=True)

raw_filename = f"{current_time}_ov{overlap:.2f}.csv"
raw_history_filename = f"{current_time}_ov{overlap:.2f}_history.csv"

raw_filepath = raw_dir / raw_filename
raw_history_filepath = raw_dir / raw_history_filename

q_points = minimize_params['q_points']
fixed_rates = np.linspace(0, overlap, q_points)
best_lagrangians = np.full(q_points, -np.inf)
best_histories = [[] for _ in range(q_points)]

with timetagger_session(500, 50, 2, [29443, 0]) as timetagger:
    with slm_session() as slm:
        experiment = Experiment(timetagger, slm, prepared_state_set, dim)
        for trial in trange(minimize_params['trial'], desc="Trials"):
            for fr_idx, fixed_rate in enumerate(fixed_rates):
                parameter_history = []


                def tracking_objective(x, *args):
                    current_lagrangian = experiment.cobyla_objective(x, *args)
                    parameter_history.append(x.copy().tolist() + [float(-current_lagrangian)])
                    return current_lagrangian


                initial_parameter = np.random.uniform(0, 2 * np.pi, size=((dim ** 2) - 1))
                result = minimize(
                    fun=tracking_objective,
                    x0=initial_parameter,
                    args=(prior_probability, fixed_rate, lambda_val),
                    **opt_config
                )
                lagrangian = -result.fun

                vector_list = unitary_matrix(result.x, dim).T

                temp_rate = np.zeros(shape=(dim - 1, dim))
                for state_idx, state in enumerate(prepared_state_set):
                    slm[0].imshow(experiment.state_holograms[str(state)])
                    for vector_idx, vector in enumerate(vector_list):
                        fields = generate_oam_superposition(
                            res=experiment.res,
                            pixel_pitch=experiment.pixel_pitch,
                            beam_w0=experiment.w0,
                            l_modes=experiment.l_modes,
                            p_modes=experiment.p_modes,
                            weights=vector.conj(),
                            prepare=True,
                            measure=False
                        )
                        projection_hologram = encode_hologram(*fields, pixel_pitch=experiment.pixel_pitch, d=8, N_steps=8,  M=1, prepare=False, measure=True, save=False)
                        slm[1].imshow(projection_hologram)

                        time.sleep(0.2)

                        count_data = timetagger.getData()
                        A_channel_counts = np.sum(a=count_data, axis=1)[0]
                        B_channel_counts = np.sum(a=count_data, axis=1)[1]
                        coincidence_data = np.sum(a=count_data, axis=1)[2]
                        coincidence_data -= A_channel_counts * B_channel_counts * cw * 1e-12
                        temp_rate[state_idx][vector_idx] += prior_probability[state_idx] * state[vector_idx]

                P_success = np.trace(temp_rate)
                P_fail = np.sum(temp_rate[:, -1])
                P_error = np.sum(temp_rate) - (P_success + P_fail)
                new_row = [overlap, trial, fixed_rate, P_success, P_error, P_fail, lagrangian, lambda_val, opt_config['method'], result.nfev]
                raw_df = pd.concat([raw_df, pd.DataFrame([new_row], columns=columns['sim data'])], ignore_index=True)

                if lagrangian > best_lagrangians[fr_idx]:
                    best_histories[fr_idx] = parameter_history
                    best_lagrangians[fr_idx] = lagrangian

history_dict = {f"{fixed_rates[i]:.3f}": pd.Series(best_histories[i]) for i in range(q_points)}
sim_history_df = pd.DataFrame(history_dict)
sim_history_df.columns = pd.MultiIndex.from_product([["fixed rate"], sim_history_df.columns])

raw_df.to_csv(raw_filepath, index=False)
sim_history_df.to_csv(raw_history_filepath, index=False)
# endregion

best_idx = raw_df['lagrangian'].idxmax()
best_fixed_rate = raw_df.loc[best_idx, 'fixed rate']
max_P_succ = raw_df.loc[best_idx, 'success rate']
theory_P_succ = theory_df.iloc[(theory_df['fixed rate'] - best_fixed_rate).abs().argsort()[:1]]['success rate'].values[0]
avg_lag = raw_df['lagrangian'].mean()
elapsed_time_raw = time.time() - start
minutes, seconds = divmod(int(elapsed_time_raw), 60)
time_str = f"{minutes}m {seconds}s"
send_message(dim, overlap, opt_config['method'], minimize_params['lambda_val'], max_P_succ, theory_P_succ, avg_lag, time_str, minimize_params['trial'], raw_filename)
