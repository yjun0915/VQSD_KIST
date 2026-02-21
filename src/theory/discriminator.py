import numpy as np
import cvxpy as cp

from scipy.optimize import minimize
from ..utils.quantum_states import get_rho_list, get_ensemble_sigma, unitary_matrix


def solve_sdp_bound(prepared_state_set, prior_probability, fixed_rate):
    """
    CVXPY를 사용하여 QSD(Quantum State Discrimination)의 이론적 확률 상한(SDP Bound)을 계산
    """
    dim = len(prepared_state_set[0])
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
    print(result)

    optimal_measurements = {}
    for con_idx, con in enumerate(cons_rho):
        if con.dual_value is not None:
            optimal_measurements[f"M{con_idx + 1}"] = con.dual_value * 2.0
    if con_sigma.dual_value is not None:
        optimal_measurements[f"M0"] = con_sigma.dual_value * 2.0

    return optimal_measurements


def _cobyla_objective(x, state_list, prior_prob_list, dim, fixed_rate, _lambda, _):
    """COBYLA 최적화에 사용될 내부 목적 함수"""
    vector_list = unitary_matrix(x, dim).T
    present_matrix = np.zeros(shape=(dim - 1, dim))

    for col, vector in enumerate(vector_list):
        for row, state in enumerate(state_list):
            prob = (np.abs(np.vdot(state, vector))) ** 2
            present_matrix[row][col] = prob

    for idx, row in enumerate(present_matrix):
        norm_factor = sum(row)
        present_matrix[idx, :] = (np.array(row)/norm_factor).tolist()

    success = np.trace(present_matrix)
    fail = np.sum(present_matrix[:, -1])

    return -(success - _lambda * (np.abs(fail - fixed_rate)))


def solve_variational_cobyla(prepared_state_set, prior_probability, Q_list, UD_starting_point, P_Success_theory,
                             trial_num=50, tolerance=0.02, _lambda=3):
    """
    주어진 fixed_rate 리스트에 대해 랜덤 파라미터 스타트를 반복하며 COBYLA 최적화를 수행합니다.
    """
    dim = len(prepared_state_set[0])

    success_point = []
    best_success_list, best_error_list, best_fail_list = [], [], []

    for qdx, Q in enumerate(tqdm(Q_list, desc="Variational COBYLA")):
        success_rate = 0
        if Q > UD_starting_point:
            break

        point_success, point_error, point_fail = [], [], []

        for trial_idx in range(trial_num):
            rand_params = np.random.uniform(0, 2 * np.pi, size=((dim ** 2) - 1))
            res = minimize(
                _cobyla_objective,
                rand_params,
                args=(prepared_state_set, prior_probability, dim, Q, _lambda, trial_idx),
                method='COBYLA',
                tol=0.01,
                options={'rhobeg': np.pi / 2, 'maxiter': 1e3, 'disp': False}
            )

            from src.utils.quantum_states import unitary_matrix
            vector_list = unitary_matrix(res.x, dim).T
            present_matrix = np.zeros(shape=(dim - 1, dim))

            for row, state in enumerate(prepared_state_set):
                raw_probs = []
                for col, vector in enumerate(vector_list):
                    prob = (np.abs(np.vdot(state, vector))) ** 2
                    raw_probs.append(prob)
                raw_probs = np.array(raw_probs)
                norm_factor = np.sum(raw_probs)
                cond_probs = raw_probs / norm_factor if norm_factor > 0 else raw_probs

                present_matrix[row, :] = cond_probs * prior_probability[row]

            p_succ = np.trace(present_matrix)
            p_fail = np.sum(present_matrix[:, -1])
            p_err = np.sum(present_matrix) - (p_succ + p_fail)

            point_success.append(p_succ)
            point_fail.append(p_fail)
            point_error.append(p_err)

            # 성공률 카운트
            if np.abs(p_succ - P_Success_theory[qdx]) < tolerance:
                success_rate += 1 / trial_num

        success_point.append(success_rate)

        # 현재 Q에서 가장 좋았던(Success가 높은) 결과만 추출해서 저장
        best_arg = np.argmax(point_success)
        best_success_list.append(point_success[best_arg])
        best_fail_list.append(point_fail[best_arg])
        best_error_list.append(point_error[best_arg])

    return success_point, best_success_list, best_fail_list, best_error_list