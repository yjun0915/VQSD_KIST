import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast, re

from pathlib import Path
from colorspacious import cspace_convert
from src.theory.error_bar import *
from matplotlib.widgets import RadioButtons

plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'axes.grid': False})


def parse_numpy_string(val):
    """'np.float64(값)' 형태의 문자열을 정규식으로 제거하고 파이썬 객체로 변환"""
    if pd.isna(val) or not isinstance(val, str):
        return val
    # np.float64(1.23) -> 1.23 으로 변환
    clean_val = re.sub(r'np\.float\d*\(', '', val).replace(')', '')
    try:
        return np.array(ast.literal_eval(clean_val))
    except:
        return val


def plot_qsd_results(script, dim, overlap, noise=0):
    # region load data
    path = Path(f"../data/{script}/dim_{dim}/ov{overlap}_noise{noise}.csv")
    theory_path = Path(f"../data/theory/dim_{dim}/ov{overlap}.csv")
    if not path or not theory_path:
        print("Cannot find data")
        return
    df_theory = pd.read_csv(theory_path)
    df = pd.read_csv(path)
    # endregion

    #region data parsing
    df['raw_data'] = df['raw_data'].apply(parse_numpy_string)
    df['history'] = df['history'].apply(ast.literal_eval)
    # endregion

    if script == 'simulation':
        df['raw_data'] = df['raw_data'].apply(lambda m: np.array(m)*100)

    # region data preprocessing
    cols = ['success rate', 'failure rate', 'error rate', 'success std', 'failure std', 'error std']
    df[cols] = df['raw_data'].apply(get_monte_carlo_error)
    df['lagrangian'] = df['success rate'] - df['lambda_val']*np.abs(df['failure rate']-df['fixed rate'])
    df['constraint_error'] = (df['lagrangian'] - df['success rate']).abs()
    # endregion

    # region data selection
    best_idx = df.groupby(['optimizer', 'fixed rate'])['constraint_error'].idxmin()
    best_df_all = df.loc[best_idx].sort_values(['optimizer', 'fixed rate']).reset_index(drop=True)
    # endregion

    # region get optimizer list
    optimizers = best_df_all['optimizer'].unique()
    if len(optimizers) == 0:
        print("옵티마이저 데이터를 찾을 수 없습니다.")
        return
    # endregion

    # region prepare canvas
    fig, (ax1, ax0) = plt.subplots(1, 2, figsize=(10.46, 6), gridspec_kw={'width_ratios': [1, 1.81]})
    ax2 = ax1.twinx()
    ax_radio = plt.axes([0.85, 0.45, 0.135, 0.15], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, optimizers)
    # endregion

    # region draw theoretical line
    ax0.plot(df_theory['fixed rate'], df_theory['success rate'], label='SDP Bound (Theory)', color='dodgerblue', linestyle='-')
    ax0.plot(df_theory['fixed rate'], df_theory['error rate'], label='SDP Bound (Theory)', color='firebrick', linestyle='-')
    ax0.plot(df_theory['fixed rate'], df_theory['failure rate'], label='SDP Bound (Theory)', color='limegreen', linestyle='-')
    # endregion

    state = {
        'current_opt': optimizers[0],
        'best_df': None,
        'current_idx': None,
        'scatters': []  # 매번 지우고 다시 그릴 scatter/errorbar 객체들
    }


    def update_main_plot(idx):
        return
    # best_df = df.loc[df.groupby('fixed rate')['lagrangian'].idxmax()]
    df['constraint_error'] = (df['lagrangian'] - df['success rate']).abs()
    best_idx = df.groupby('fixed rate')['constraint_error'].idxmin()
    best_df = df.loc[best_idx]

    ax0.errorbar(best_df['fixed rate'], best_df['success rate'], yerr=best_df['success std'].to_numpy(), fmt='o', color='dodgerblue', ecolor='dodgerblue', elinewidth=1, capsize=2, label='Success (MC Error)')
    ax0.errorbar(best_df['fixed rate'], best_df['error rate'], yerr=best_df['error std'].to_numpy(), fmt='o', color='firebrick', ecolor='firebrick', elinewidth=1, capsize=2, label='Success (MC Error)')
    ax0.errorbar(best_df['fixed rate'], best_df['failure rate'], yerr=best_df['failure std'].to_numpy(), fmt='o', color='limegreen', ecolor='limegreen', elinewidth=1, capsize=2, label='Success (MC Error)')

    x_data = best_df['fixed rate'].to_numpy()
    y_succ = best_df['success rate'].to_numpy()
    y_err = best_df['error rate'].to_numpy()
    y_fail = best_df['failure rate'].to_numpy()

    p_succ, = ax0.plot(x_data, y_succ, 'o', color='dodgerblue', label='VQE Best Result')
    p_err, = ax0.plot(x_data, y_err, 'o', color='firebrick', label='VQE Best Result')
    p_fail, = ax0.plot(x_data, y_fail, 'o', color='limegreen', label='VQE Best Result')

    ax0.set_xlabel('Fixed Rate')
    ax0.set_ylabel('Probability')
    ax0.set_title(f'Quantum State Discrimination (Dim={dim}, Overlap={overlap})')
    # ==========================================
    # 🌟 인터랙티브 로직 (Hover & Click)
    # ==========================================

    # History 데이터 미리 로드
    df_hist = df['history']

    # 강조용 빈 마커 생성
    highlight, = ax0.plot([], [], 'o', markeredgecolor='black', markerfacecolor='none',
                          markeredgewidth=2, markersize=12, zorder=10, visible=False)

    hover_state = {'current_idx': None}


    def update_trajectory(idx):
        if df_hist is None:
            return

        # 기존에 그려진 그래프 초기화 (clear를 하면 label 등도 지워지므로 다시 세팅 필요)
        ax1.clear()
        ax2.clear()

        # 데이터 파싱
        trajectory_list = list(map(list, zip(*df_hist[idx])))
        if not trajectory_list:
            return

        trajectory = np.array(trajectory_list)

        num_params = trajectory.shape[1] - 1
        colors = [cspace_convert([70, 60, 360 * (theta / num_params)], "CIELCh", "sRGB1") for theta in
                  range(num_params)]
        colors = np.clip(colors, 0, 1)

        for i in range(int(dim * (dim - 1) / 2)):
            ax1.plot(trajectory[:, i], label=rf'$\theta_{i + 1}$', color=colors[i], linewidth=1)
        for i in range(int(dim * (dim - 1) / 2), (dim ** 2) - 1):
            ax1.plot(trajectory[:, i], label=rf'$\varphi_{i - int(dim * (dim - 1) / 2) + 1}$', color=colors[i],
                     linewidth=1)

        ax1.set_xlabel('Optimization Iterations')
        ax1.set_ylabel('Parameter Value (rad)', color='k')
        Q = best_df.iloc[idx]['fixed rate']
        ax1.set_title(f'Trajectory (fixed rate = {Q})')

        ax2.set_ylabel('Lagrangian Value', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        ax2.plot(trajectory[:, -1], label=r'$\mathcal{L}$', color='firebrick', linewidth=1.5)

        fig.canvas.draw_idle()

    def on_hover(event):
        if event.inaxes != ax0:
            if highlight.get_visible():
                highlight.set_visible(False)
                hover_state['current_idx'] = None
                fig.canvas.draw_idle()
            return

        is_hovered = False
        for line in [p_succ, p_err, p_fail]:
            cont, ind_dict = line.contains(event)
            if cont:
                is_hovered = True
                idx = ind_dict['ind'][0]

                if hover_state['current_idx'] != idx:
                    target_x = [x_data[idx], x_data[idx], x_data[idx]]
                    target_y = [y_succ[idx], y_err[idx], y_fail[idx]]

                    highlight.set_data(target_x, target_y)
                    highlight.set_visible(True)
                    hover_state['current_idx'] = idx
                    fig.canvas.draw_idle()
                break

        if not is_hovered and highlight.get_visible():
            highlight.set_visible(False)
            hover_state['current_idx'] = None
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax0 and event.button == 1:  # 1은 마우스 좌클릭
            idx = hover_state['current_idx']
            if idx is not None:
                print(f"🖱️ Clicked Fixed Rate index: {idx}, Updating Trajectory...")
                update_trajectory(idx)

    update_main_plot(0)
    update_trajectory(0)

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    radio.on_clicked(update_main_plot)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_qsd_results(script='experiment', dim=3, overlap=0.75, noise=0)
