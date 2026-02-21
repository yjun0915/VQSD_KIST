import numpy as np


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
    d_params = params[(n*(n-1)):((n**2)-1)]

    t_matrix = np.eye(n, dtype=complex)
    d_matrix = np.eye(n, dtype=complex)

    s_r = [i-j for i in range(1, n) if i % 2 == 1 for j in range(0, i)]
    s_l = [n+j-i-1 for i in range(1, n) if i % 2 == 0 for j in range(1, i+1)]
    s = [*s_l, *s_r[::-1]]

    for s_idx, order in enumerate(s):
        bs = np.array([
            [np.exp(1j*phi)*np.cos(theta[s_idx]), -np.sin(theta[s_idx])],
            [np.exp(1j*phi)*np.sin(theta[s_idx]), np.cos(theta[s_idx])],
        ])
        t_component = np.eye(n, dtype=complex)
        t_component[order-1:order+1, order-1:order+1] = bs
        t_matrix = t_matrix@t_component

    for idx in range(n):
        d_matrix[idx, idx] = np.exp(1j*d_params[idx])

    matrix = d_matrix@t_matrix

    return matrix
