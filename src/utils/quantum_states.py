import numpy as np
from scipy.optimize import minimize


def get_rho_list(states):
    """상태 벡터 리스트를 밀도 행렬(rho) 리스트로 변환합니다."""
    return [np.outer(s, s.conj()) for s in states]


def get_ensemble_sigma(prior_probabilities, rho_list):
    """주어진 사전 확률과 밀도 행렬 리스트로부터 평균 밀도 행렬(sigma)을 계산합니다."""
    return sum(p * r for p, r in zip(prior_probabilities, rho_list))


def unitary_matrix(params, n):
    """
    파라미터를 유니터리 행렬로 디자인.
    William R. Clements, et al. , "Optimal design for universal multiport interferometers," Optica 3, 1460-1465 (2016)
    """
    theta = params[0:int(n*(n-1)/2)]
    phi = params[int(n*(n-1)/2):(n*(n-1))]
    d_params = [0] + list(params[(n*(n-1)):((n**2)-1)])

    t_matrix = np.eye(n, dtype=complex)
    d_matrix = np.eye(n, dtype=complex)

    s_r = [i-j for i in range(1, n) if i % 2 == 1 for j in range(0, i)]
    s_l = [n+j-i-1 for i in range(1, n) if i % 2 == 0 for j in range(1, i+1)]
    s = [*s_l, *s_r[::-1]]

    for s_idx, order in enumerate(s):
        bs = np.array([
            [np.exp(1j*phi[s_idx])*np.cos(theta[s_idx]), -np.sin(theta[s_idx])],
            [np.exp(1j*phi[s_idx])*np.sin(theta[s_idx]), np.cos(theta[s_idx])],
        ])
        t_component = np.eye(n, dtype=complex)
        t_component[order-1:order+1, order-1:order+1] = bs
        t_matrix = t_matrix@t_component

    for idx in range(n):
        d_matrix[idx, idx] = np.exp(1j*d_params[idx])

    matrix = d_matrix@t_matrix

    return matrix


def get_discrimination_rates(state, measure, prior_probability):
    P_success = 0
    P_error = 0
    P_fail = 0
    for state_idx, state in enumerate(state):
        prob = prior_probability[state_idx]
        for povm_idx in range(len(measure)):
            measurement = measure[f"M{povm_idx}"]
            rho = state if np.array(state).ndim == 2 else np.outer(state, np.conj(state))
            M = measurement if measurement.ndim == 2 else np.outer(measurement, np.conj(measurement))
            if povm_idx == 0:
                P_fail += prob * np.real(np.trace(rho @ M))
            elif povm_idx - state_idx == 1:
                P_success += prob * np.real(np.trace(rho @ M))
            else:
                P_error += prob * np.real(np.trace(rho @ M))
    return P_success, P_error, P_fail


def prepared_state_d_dim(dim, overlap):
    if dim < 2:
        return None

    target_gram_matrix = np.empty(shape=(dim, dim))
    target_gram_matrix.fill(overlap)
    for idx in range(dim): target_gram_matrix[idx, idx] = 1

    def obj(x):
        vectors = np.empty(shape=(dim, dim))
        for d in range(dim):
            vectors[d, 0] = np.cos(x[d*(dim-1)])
            vectors[d, 1] = np.sin(x[d*(dim-1)])
        for m in range(dim-2):
            for d in range(dim):
                vectors[d, 2+m] = vectors[d, 1+m]*np.sin(x[(d * (dim - 1)) + 1 + m])
                vectors[d, 1+m] = vectors[d, 1+m]*np.cos(x[(d * (dim - 1)) + 1 + m])

        gram_matrix = np.empty(shape=(dim, dim))
        for row, vector_row in enumerate(vectors):
            for col, vector_col in enumerate(vectors):
                gram_matrix[row, col] = np.vdot(vector_col, vector_row)

        return np.trace(np.dot((target_gram_matrix-gram_matrix).T, (target_gram_matrix-gram_matrix)))
    init = np.random.uniform(0, 2*np.pi, size=(dim*(dim-1)))
    result = minimize(obj, init, method='COBYLA', tol=1e-7, options={'maxiter': 10000, 'rhobeg':1.570796})

    output_vectors = np.empty(shape=(dim, dim))
    for d in range(dim):
        output_vectors[d, 0] = np.cos(result.x[d * (dim - 1)])
        output_vectors[d, 1] = np.sin(result.x[d * (dim - 1)])
    for m in range(dim - 2):
        for d in range(dim):
            output_vectors[d, 2 + m] = output_vectors[d, 1 + m] * np.sin(result.x[(d * (dim - 1)) + 1 + m])
            output_vectors[d, 1 + m] = output_vectors[d, 1 + m] * np.cos(result.x[(d * (dim - 1)) + 1 + m])
    return np.array(output_vectors)
