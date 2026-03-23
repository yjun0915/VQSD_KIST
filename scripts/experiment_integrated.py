import time
import screeninfo

import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

from TimeTagger import createTimeTagger, Resolution_Standard, Coincidences, Counter
from tqdm import trange
from scipy.optimize import minimize

from src.utils.quantum_states import *
from OAM_KIST.holography import *


dim = 3
overlap = 0.75
prior_probability = [1/(dim-1) for _ in range(dim-1)] + [0]
prepared_state_set = np.hstack((prepared_state_d_dim(dim-1, overlap), [[0] for _ in range(dim-1)]))
rho_list = get_rho_list(prepared_state_set)

cw, binwidth, n_value = 50, 50, 2
integration_time = binwidth*n_value/1000

opt_config = {'method': "COBYLA", "tol": 0.01}
lambda_val = 1
fixed_rates = [0, 0.1875, 0.3750, 0.5625, 0.75]
spacing = 2

dir = Path(f"../data/experiment/dim_{dim}")
dir.mkdir(parents=True, exist_ok=True)
filename = f"ov{overlap:.2f}_noise0.csv"
filename_hist = f"hist_ov{overlap:.2f}_noise0.csv"
filepath = dir / filename
filepath_hist = dir / filename_hist
is_new_file = not filepath.exists()
is_new_hist = not filepath_hist.exists()

# region timetagger
tagger = createTimeTagger(resolution=Resolution_Standard)

coincidences = Coincidences(
    tagger=tagger,
    coincidenceGroups=[[1, 2]],
    coincidenceWindow=cw,
)
counter = Counter(
    tagger=tagger,
    channels=[1, 2, list(coincidences.getChannels())[0]],
    binwidth=binwidth * 1e9,
    n_values=n_value,
)
tagger.setInputDelay(channel=1, delay=29443)
tagger.setInputDelay(channel=2, delay=0)
# endregion

# region SLM
class Fullscreen_CV():
    delay: int = 200  # internal delay time after imshow
    # screen_id 기본값 = 0
    def __init__(self, screen_id: int = 0):
        # screen_id
        self.monitor = screeninfo.get_monitors()[screen_id]
        self.width = self.monitor.width
        self.height = self.monitor.height
        self.x = self.monitor.x
        self.y = self.monitor.y
        # screen_id
        self.name = str(screen_id)

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, self.x, self.y)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_TOPMOST, 1)

        # set initial image
        img_gray = np.full((self.height, self.width), 127, dtype=np.uint8)
        self.imshow(img_gray)
        cv2.waitKey(1900)  # first imshow require long delay time

    @property
    def shape(self):
        return self.height, self.width, 3

    def imshow(self, image: np.ndarray):
        cv2.imshow(self.name, image)
        cv2.waitKey(self.delay)

    def destroyWindow(self):
        cv2.destroyWindow(self.name)

slm1 = Fullscreen_CV(1)
slm2 = Fullscreen_CV(2)
# endregion

# region OAM pre-processing
l_modes = [-1]
for d in np.arange(2, dim + 1):
    if (d ^ 1) & 1:
        l_modes = l_modes + [int(d / 2) * spacing - 1]
    elif d & 1:
        l_modes = [-(int((d - 1) / 2 * spacing + 1))] + l_modes
p_modes = np.zeros_like(l_modes)
if spacing == 2:
    for idx, l in enumerate(l_modes):
        p_modes[idx] = 9 - np.abs(l)
        if np.abs(l) == 7:
            p_modes += 1

state_holograms = {}
for state in prepared_state_set:
    fields = generate_oam_superposition(
        res=[1920, 1080],
        pixel_pitch=8e-6,
        beam_w0=0.8e-3,
        l_modes=l_modes,
        p_modes=p_modes,
        weights=state,
        prepare=True,
        measure=False
    )
    state_holograms[str(state)] = encode_hologram(*fields, pixel_pitch=8e-6, d=8, N_steps=8, M=1, prepare=True, measure=False, save=False)
# endregion

start = time.time()
for fr_idx, fixed_rate in enumerate(fixed_rates):
    parameter_history = []
    def obj(x):
        vector_list = unitary_matrix(x, dim).T

        temp_rate = np.zeros(shape=(dim - 1, dim))
        for state_idx, state in enumerate(prepared_state_set):
            slm1.imshow(state_holograms[str(state)])
            for vector_idx, vector in enumerate(vector_list):
                fields = generate_oam_superposition(
                    res=[1920, 1080],
                    pixel_pitch=8e-6,
                    beam_w0=0.8e-3,
                    l_modes=l_modes,
                    p_modes=p_modes,
                    weights=vector.conj(),
                    prepare=True,
                    measure=False
                )
                projection_hologram = encode_hologram(*fields, pixel_pitch=8e-6, d=8, N_steps=8, M=1, prepare=False, measure=True, save=False)
                slm2.imshow(projection_hologram)

                counter.clear()
                time.sleep(integration_time)

                count_data = counter.getData()
                A_channel_counts = np.sum(a=count_data, axis=1)[0]
                B_channel_counts = np.sum(a=count_data, axis=1)[1]
                coincidence_data = np.sum(a=count_data, axis=1)[2]
                coincidence_data -= max(0, A_channel_counts * B_channel_counts * cw * 1e-12)
                temp_rate[state_idx][vector_idx] += prior_probability[state_idx]*coincidence_data

        total_count = np.sum(temp_rate)
        if total_count > 0:
            temp_rate /= total_count

        success = np.trace(temp_rate)
        failure = np.sum(temp_rate[:, -1])

        return -(success - lambda_val*np.abs(failure - fixed_rate))

    def callback(xk, intermediate_result):
        parameter_history.append(xk.copy().tolist() + [float(-intermediate_result.fun)])
        return 0

    x0 = np.random.uniform(0, 2 * np.pi, size=((dim ** 2) - 1))
    minimize_result = minimize(
        fun=obj,
        x0=x0,
        method=opt_config['method'],
        options={
            'tol': opt_config['tol'],
            'callback': callback,
        }
    )

    print(minimize_result)

    raw_data = [[0 for __ in range(dim)] for _ in range(dim-1)]
    vector_list = unitary_matrix(minimize_result.x, dim).T
    for state_idx, state in enumerate(prepared_state_set):
        slm1.imshow(state_holograms[str(state)])
        for vector_idx, vector in enumerate(vector_list):
            fields = generate_oam_superposition(
                res=[1920, 1080],
                pixel_pitch=8e-6,
                beam_w0=0.8e-3,
                l_modes=l_modes,
                p_modes=p_modes,
                weights=vector.conj(),
                prepare=True,
                measure=False
            )
            projection_hologram = encode_hologram(*fields, pixel_pitch=8e-6, d=8, N_steps=8, M=1, prepare=False, measure=True, save=False)
            slm2.imshow(projection_hologram)

            counter.clear()
            time.sleep(integration_time)

            count_data = counter.getData()
            A_channel_counts = np.sum(a=count_data, axis=1)[0]
            B_channel_counts = np.sum(a=count_data, axis=1)[1]
            coincidence_data = np.sum(a=count_data, axis=1)[2]
            coincidence_data -= max(0, A_channel_counts * B_channel_counts * cw * 1e-12)
            raw_data[state_idx][vector_idx] += float(prior_probability[state_idx] * coincidence_data)
    current_time = datetime.now().strftime("%y%m%d%H%M%S")
    new_row_df = pd.DataFrame([{
        'timestamp': current_time,
        'optimizer': opt_config['method'],
        'lambda_val': lambda_val,
        'fixed rate': fixed_rate,
        'raw_data': raw_data
    }])
    new_hist_df = pd.DataFrame([{
        'timestamp': current_time,
        'optimizer': opt_config['method'],
        'lambda_val': lambda_val,
        'fixed rate': fixed_rate,
        'history': list(map(list, zip(*parameter_history)))
    }])
    new_row_df.to_csv(filepath, mode='a', index=False, header=is_new_file, encoding='utf-8-sig')
    new_hist_df.to_csv(filepath_hist, mode='a', index=False, header=is_new_hist, encoding='utf-8-sig')
    is_new_file = False
    is_new_hist = False
end = time.time()
elapsed_time_raw = end - start
minutes, seconds = divmod(elapsed_time_raw, 60)
time_str = f"{int(minutes)}m {seconds:.2f}s"
print(time_str)
