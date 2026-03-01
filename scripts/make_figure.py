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

    plt.figure(figsize=(8, 6))

    plt.plot(df_theory['fixed rate'], df_theory['success rate'], label='SDP Bound (Theory)', color='dodgerblue', linestyle='-')
    plt.plot(df_theory['fixed rate'], df_theory['error rate'], label='SDP Bound (Theory)', color='firebrick', linestyle='-')
    plt.plot(df_theory['fixed rate'], df_theory['failure rate'], label='SDP Bound (Theory)', color='limegreen', linestyle='-')

    best_sim = df_sim.loc[df_sim.groupby('fixed rate')['lagrangian'].idxmax()]

    plt.plot(best_sim['fixed rate'], best_sim['success rate'], 'o', color='dodgerblue', label='VQE Best Result')
    plt.plot(best_sim['fixed rate'], best_sim['error rate'], 'o', color='firebrick', label='VQE Best Result')
    plt.plot(best_sim['fixed rate'], best_sim['failure rate'], 'o', color='limegreen', label='VQE Best Result')

    plt.xlabel('Fixed Rate (Error or Failure Constraint)')
    plt.ylabel('Success Probability')
    plt.title(f'Quantum State Discrimination (Dim={dim}, Overlap={overlap})')
    plt.legend()
    # plt.show()

    if history_path:
        df_hist = pd.read_csv(history_path)

        col_name = df_hist.columns[0]

        trajectory = df_hist[col_name].dropna().apply(ast.literal_eval).tolist()
        selected_fixed_rate = trajectory[0]
        trajectory = np.array(trajectory[1:])

        fig, ax1 = plt.subplots(figsize=(7, 6))
        num_params = trajectory.shape[1] - 1
        dim = np.sqrt(num_params + 1).astype(int)
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
        ax1.legend(lines, labels, bbox_to_anchor=(1.2, 1), loc='upper left')

        plt.xlabel('Optimization Iterations')
        plt.ylabel('Parameter Value (rad)')
        plt.title(f'Trajectory of Best Trial (@ fixed rate = {selected_fixed_rate})')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_qsd_results(dim=3, overlap=0.75)
