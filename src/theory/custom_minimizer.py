import numpy as np
from scipy.optimize import OptimizeResult


class Adam:
    """Adam Optimizer for SciPy compatibility"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.iter = 0

    def get_grad(self, fun, x, args, jac):
        """Gradient를 계산하는 내부 함수"""
        if callable(jac):
            # 1. 해석적 미분 (Parameter Shift Rule 등 사용 시)
            return np.array(jac(x, *args))
        else:
            # 2. 수치 미분 (Finite Difference): jac가 제공되지 않았을 때
            eps = 1e-8
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                # 중앙 차분법 (Central Difference)
                grad[i] = (fun(x_plus, *args) - fun(x_minus, *args)) / (2 * eps)
            return grad

    def adam(self, fun, x0, args=(), jac=None, maxiter=1000, tol=1e-2, **kwargs):
        # 파라미터를 1D Numpy Array로 관리
        x = np.array(x0, dtype=float)

        # 모멘텀 초기화
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        history = []  # 🌟 VQE 궤적 저장을 위한 리스트

        for i in range(maxiter):
            self.iter += 1

            # 1. 목적 함수 값 계산 및 히스토리 저장
            current_val = fun(x, *args)
            history.append(x.tolist() + [float(current_val)])  # 이전 대화에서 수정한 float() 적용

            # 2. 그래디언트 계산
            g = self.get_grad(fun, x, args, jac)

            # 3. 종료 조건 확인
            if np.linalg.norm(g) < tol:
                print(f"Optimization converged at iteration {self.iter}")
                break

            # 4. Adam 업데이트 (Vectorized 연산으로 한 번에 처리)
            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)

            # Bias correction (편향 보정)
            m_hat = self.m / (1 - self.beta1 ** self.iter)
            v_hat = self.v / (1 - self.beta2 ** self.iter)

            # 파라미터 업데이트
            x -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # SciPy OptimizeResult 형태로 결과 반환
        result = OptimizeResult()
        result.x = x
        result.fun = current_val
        result.jac = g
        result.nit = self.iter
        result.success = (self.iter < maxiter)
        result.message = "Optimization terminated successfully." if result.success else "Maximum number of iterations reached."
        result.history = history  # 🌟 커스텀 속성 추가

        return result