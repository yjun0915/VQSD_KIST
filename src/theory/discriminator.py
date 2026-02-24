import time

import numpy as np
import cvxpy as cp

from ..utils.quantum_states import *
from OAM_KIST.holography import *


cw = 500

def solve_sdp_bound(prepared_state_set, prior_probability, dim, fixed_rate):
    """
    CVXPY를 사용하여 QSD(Quantum State Discrimination)의 이론적 확률 상한(SDP Bound)을 계산
    """
    rho_list = get_rho_list(prepared_state_set)
    sigma = get_ensemble_sigma(prior_probability, rho_list)

    Lambda = cp.Variable((dim, dim), hermitian=True)
    a = cp.Variable(nonneg=True)

    constraints = []

    cons_rho = []
    for prob, rho in zip(prior_probability, rho_list):
        con = (Lambda - prob * rho >> 0)
        cons_rho.append(con)
        constraints.append(con)

    con_sigma = (Lambda - a * sigma >> 0)
    constraints.append(con_sigma)

    # 1번째 목적 함수: Trace(Lambda) - a * fixed_rate 최소화
    objective = cp.Minimize(cp.trace(Lambda) - a * fixed_rate)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS, eps=1e-6)

    optimal_measurements = {}
    for con_idx, con in enumerate(cons_rho):
        if con.dual_value is not None:
            optimal_measurements[f"M{con_idx + 1}"] = con.dual_value * 2.0
    if con_sigma.dual_value is not None:
        optimal_measurements[f"M0"] = con_sigma.dual_value * 2.0

    return optimal_measurements


def cobyla_objective(x, state_list, prior_prob_list, dim, fixed_rate, _lambda):
    """COBYLA 최적화에 사용될 내부 목적 함수"""
    vector_list = unitary_matrix(x, dim).T

    measurements = {}
    for vector_idx, vector in enumerate(vector_list):
        measurements[f"M{np.mod(vector_idx + 1, dim)}"] = vector

    success, error, fail = get_discrimination_rates(state_list, measurements, prior_prob_list)

    return -(success - _lambda * (np.abs(fail - fixed_rate)))


class Experiment:
    def __init__(self, timetagger, slm, state_list, dim):
        self.timetagger = timetagger
        self.slm_prepare = slm[0]
        self.slm_measure = slm[1]
        self.state_list = state_list
        self.dim = dim
        self.res = [1920, 1080]
        self.pixel_pitch = 8e-6
        self.w0 = 8e-4

        spacing = 2
        pos = np.arange(int(np.ceil(spacing / 2)), int(np.ceil(spacing / 2)) + (dim // 2) * spacing, spacing)
        self.l_modes = np.concatenate([-pos[::-1], pos]).tolist()
        self.p_modes = np.zeros_like(self.l_modes)

        self.state_holograms = {}
        for state in state_list:
            fields = generate_oam_superposition(
                res=self.res,
                pixel_pitch=self.pixel_pitch,
                beam_w0=self.w0,
                l_modes=self.l_modes,
                p_modes=self.p_modes,
                weights=state,
                prepare=True,
                measure=False
            )
            # encode_hologram(*fields, pixel_pitch=pixel_pitch, d=8, N_steps=8, M=1, prepare=True, measure=False,
            #                 save=True, path="temporal_images", name=str(state))
            self.state_holograms[str(state)] = encode_hologram(*fields, pixel_pitch=self.pixel_pitch, d=8, N_steps=8,
                                                               M=1, prepare=True, measure=False, save=False)


    def cobyla_objective(self, x, prior_prob_list, fixed_rate, _lambda):
        """COBYLA 최적화에 사용될 내부 목적 함수
        SLM A 에서 0.2초 마다 1/d 확률로 랜덤한 상태를 띄우고, SLM B 에서 이걸 모르는 상태로 측정
        측정 시간만 0.2*d 이상으로 (아마 정수배) 준비 -> 0.2초 측정을 d*n 회 하는것"""
        vector_list = unitary_matrix(x, self.dim).T

        temp_rate = np.zeros(shape=(self.dim-1, self.dim))
        for state_idx, state in enumerate(self.state_list):
            self.slm_prepare.imshow(self.state_holograms[str(state)])
            for vector_idx, vector in enumerate(vector_list):
                fields = generate_oam_superposition(
                    res=self.res,
                    pixel_pitch=self.pixel_pitch,
                    beam_w0=self.w0,
                    l_modes=self.l_modes,
                    p_modes=self.p_modes,
                    weights=vector.conj(),
                    prepare=True,
                    measure=False
                )
                projection_hologram = encode_hologram(*fields, pixel_pitch=self.pixel_pitch, d=8, N_steps=8, M=1,
                                                      prepare=False, measure=True, save=False)
                self.slm_measure.imshow(projection_hologram)

                time.sleep(0.2)

                count_data = self.timetagger.getData()
                A_channel_counts = np.sum(a=count_data, axis=1)[0]
                B_channel_counts = np.sum(a=count_data, axis=1)[1]
                coincidence_data = np.sum(a=count_data, axis=1)[2]
                coincidence_data -= A_channel_counts * B_channel_counts * cw * 1e-12
                temp_rate[state_idx][vector_idx] += prior_prob_list[state_idx]*state[vector_idx]

        success = np.trace(temp_rate)
        fail = np.sum(temp_rate[:, -1])

        return -(success - _lambda * (np.abs(fail - fixed_rate)))
