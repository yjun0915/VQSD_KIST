import numpy as np
import pandas as pd


def get_monte_carlo_error(raw_data, trials=1000):
    raw_data = np.nan_to_num(np.array(raw_data))
    raw_data = np.maximum(0, raw_data)

    # 2. 포아송 샘플링을 위한 기대값(lambda) 계산
    mean_counts = raw_data * raw_data
    dim = raw_data.shape[1]
    samples = np.random.poisson(raw_data, size=(trials, *raw_data.shape))

    success_rates = []
    failure_rates = []
    error_rates = []
    for s in samples:
        total_count = np.sum(s)
        s = s/total_count
        succ = np.trace(s)
        fail = np.sum(s[:, -1])
        err = np.sum(s) - (succ + fail)
        success_rates.append(succ)
        failure_rates.append(fail)
        error_rates.append(err)
    P_succ = np.mean(success_rates)
    P_fail = np.mean(failure_rates)
    P_err = np.mean(error_rates)
    std_succ = np.std(success_rates)
    std_fail = np.std(failure_rates)
    std_err = np.std(error_rates)
    return pd.Series({
        'success rate': P_succ,
        'failure rate': P_fail,
        'error rate': P_err,
        'success std': std_succ,
        'failure std': std_fail,
        'error std': std_err,
    })
