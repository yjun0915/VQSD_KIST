import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from pathlib import Path
from colorspacious import cspace_convert

plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'axes.grid': False})


def load_latest_file(directory, pattern="*.csv"):
    files = list(Path(directory).glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def plot_qsd_results(dim, overlap):
    sim_dir = Path(f"../data/simulation/dim_{dim}")
    theory_path = Path(f"../data/theory/dim_{dim}/ov{overlap}.csv")

    sim_path = load_latest_file(sim_dir, f"*_ov{overlap}.csv")
    history_path = load_latest_file(sim_dir, f"*_ov{overlap}_history.csv")

    if not sim_path or not theory_path:
        print("Cannot find data")
        return

    df_theory = pd.read_csv(theory_path)
    df_sim = pd.read_csv(sim_path)

    fig, (ax1, ax0) = plt.subplots(1, 2, figsize=(10.46, 6), gridspec_kw={'width_ratios': [1, 1.81]})

    ax0.plot(df_theory['fixed rate'], df_theory['success rate'], label='SDP Bound (Theory)', color='dodgerblue', linestyle='-')
    ax0.plot(df_theory['fixed rate'], df_theory['error rate'], label='SDP Bound (Theory)', color='firebrick', linestyle='-')
    ax0.plot(df_theory['fixed rate'], df_theory['failure rate'], label='SDP Bound (Theory)', color='limegreen', linestyle='-')

    best_sim = df_sim.loc[df_sim.groupby('fixed rate')['lagrangian'].idxmax()]

    ax0.plot(best_sim['fixed rate'], best_sim['success rate'], 'o', color='dodgerblue', label='VQE Best Result')
    ax0.plot(best_sim['fixed rate'], best_sim['error rate'], 'o', color='firebrick', label='VQE Best Result')
    ax0.plot(best_sim['fixed rate'], best_sim['failure rate'], 'o', color='limegreen', label='VQE Best Result')

    ax0.set_xlabel('Fixed Rate (Error or Failure Constraint)')
    ax0.set_ylabel('Success Probability')
    ax0.set_title(f'Quantum State Discrimination (Dim={dim}, Overlap={overlap})')
    ax0.legend()

    if history_path:
        df_hist = pd.read_csv(history_path)

        col_name = df_hist.columns[2]

        trajectory = df_hist[col_name].dropna().apply(ast.literal_eval).tolist()
        selected_fixed_rate = trajectory[0]
        trajectory = np.array(trajectory[1:])

        num_params = trajectory.shape[1] - 1
        colors = [cspace_convert([70, 60, 360*(theta/num_params)], "CIELCh", "sRGB1") for theta in range(num_params)]
        colors = np.clip(colors, 0, 1)

        lines = []
        for i in range(int(dim*(dim-1)/2)):
            line = ax1.plot(trajectory[:, i], label=rf'$\theta_{i + 1}$', color=colors[i], linewidth=1)
            lines += line

        for i in range(int(dim*(dim-1)/2), (dim**2)-1):
            line = ax1.plot(trajectory[:, i], label=rf'$\varphi_{i - int(dim*(dim-1)/2) + 1}$', color=colors[i], linewidth=1)
            lines += line
        ax1.set_xlabel('Optimization Iterations')
        ax1.set_ylabel('Parameter Value (rad)', color='k')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Lagrangian Value', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        line_lag = ax2.plot(trajectory[:, -1], label=r'$\mathcal{L}$', color='firebrick', linewidth=1.5)
        lines += line_lag
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, bbox_to_anchor=(0.65, 1), loc='upper left', framealpha=0.7)

        ax1.set_xlabel('Optimization Iterations')
        ax1.set_ylabel('Parameter Value (rad)')
        ax1.set_title(f'Trajectory (fixed rate = {selected_fixed_rate})')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_qsd_results(dim=4, overlap=0.75)
