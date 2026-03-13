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


# region parameter configuration
with open("../config/params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

opt_config = config['optimization'][config['minimize']['optimizer']]

lambda_val = config['minimize']['lambda_val']

minimize_params = config['minimize']

columns = config['columns']
# endregion

dim = minimize_params['dim']
overlap = minimize_params['overlap']
prior_probability = [1/(dim-1) for _ in range(dim-1)] + [0]
prepared_state_set = np.hstack((prepared_state_d_dim(dim-1, overlap), [[0] for _ in range(dim-1)]))
rho_list = get_rho_list(prepared_state_set)

# region theory data
theory_dir = Path(f"../data/theory/dim_{dim}")
theory_dir.mkdir(parents=True, exist_ok=True)
theory_filename = f"ov{overlap:.2f}.csv"
theory_filepath = theory_dir / theory_filename
theory_rows = []
for fixed_rate in np.linspace(0, 1, 100):
    optimal_measurements = solve_sdp_bound(prepared_state_set, prior_probability, dim, fixed_rate)
    P_success, P_error, P_fail = get_discrimination_rates(rho_list, optimal_measurements, prior_probability)
    theory_rows.append([overlap, fixed_rate, P_success, P_error, P_fail])
theory_df = pd.DataFrame(theory_rows, columns=columns['theory'])
theory_df.to_csv(theory_filepath, index=False)
# endregion


start = time.time()
# region experiment data
dir = Path(f"../data/experiment/dim_{dim}")
dir.mkdir(parents=True, exist_ok=True)
filename = f"ov{overlap:.2f}.csv"
filepath = dir / filename
is_new_file = not filepath.exists()

fixed_rates = minimize_params['q_points']
timetagger_config = config['devices']['timetagger']
cw, binwidth, n_value, delay = timetagger_config['cw'], timetagger_config['binwidth'], timetagger_config['n_value'], timetagger_config['delay']
with timetagger_session(cw, binwidth, n_value, delay) as timetagger:
    with slm_session() as slms:
        experiment = Experiment(timetagger, slms, prepared_state_set, dim)
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

                raw_data = [[0 for __ in range(dim)] for _ in range(dim-1)]
                vector_list = unitary_matrix(result.x, dim).T
                for state_idx, state in enumerate(prepared_state_set):
                    slms[0].imshow(experiment.state_holograms[str(state)])
                    for vector_idx, vector in enumerate(vector_list):
                        fields = generate_oam_superposition(
                            res=experiment.slm_config['res'],
                            pixel_pitch=experiment.slm_config['pixel_pitch'],
                            beam_w0=experiment.slm_config['beam_w0'],
                            l_modes=experiment.l_modes,
                            p_modes=experiment.p_modes,
                            weights=vector.conj(),
                            prepare=True,
                            measure=False
                        )
                        projection_hologram = encode_hologram(*fields, pixel_pitch=experiment.slm_config['pixel_pitch'], d=8, N_steps=8,  M=1, prepare=False, measure=True, save=False)
                        slms[1].imshow(projection_hologram)

                        time.sleep(0.2)

                        count_data = timetagger.getData()
                        A_channel_counts = np.sum(a=count_data, axis=1)[0]
                        B_channel_counts = np.sum(a=count_data, axis=1)[1]
                        coincidence_data = np.sum(a=count_data, axis=1)[2]
                        coincidence_data -= max(0, A_channel_counts * B_channel_counts * cw * 1e-12)
                        raw_data[state_idx][vector_idx] += prior_probability[state_idx] * coincidence_data

                current_time = datetime.now().strftime("%y%m%d%H%M%S")
                new_row_df = pd.DataFrame([{
                    'timestamp': current_time,
                    'optimizer': config['minimize']['optimizer'],
                    'lambda_val': lambda_val,
                    'fixed rate': fixed_rate,
                    'history': list(map(list, zip(*parameter_history))),
                    'raw_data': raw_data
                }])
                new_row_df.to_csv(filepath, mode='a', index=False, header=is_new_file, encoding='utf-8-sig')
                is_new_file = False
# endregion
end = time.time()
elapsed_time_raw = end - start
minutes, seconds = divmod(elapsed_time_raw, 60)
time_str = f"{int(minutes)}m {seconds:.2f}s"
