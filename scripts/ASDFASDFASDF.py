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
        df['raw_data'] = df['raw_data'].apply(lambda m: np.array(m)*1000)

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

    #########################################
    # 🌟 상태 관리 및 메인 플롯 업데이트 로직
    #########################################
    state = {
        'current_opt': optimizers[0],
        'best_df': None,
        'current_idx': None,
        'scatters': [],  # Hover 감지용 투명 마커들
        'highlight': None  # 클릭/호버 시 강조되는 마커
    }

    def update_main_plot(opt_name):
        state['current_opt'] = opt_name

        # 1. 해당 옵티마이저 데이터만 필터링 후 인덱스 리셋
        cur_df = best_df_all[best_df_all['optimizer'] == opt_name].reset_index(drop=True)
        state['best_df'] = cur_df

        # 2. 우측 그래프(ax0) 초기화
        ax0.clear()

        # 3. 이론값(Theory) 다시 그리기 (clear 되면 지워지므로 여기서 매번 그려야 함)
        ax0.plot(df_theory['fixed rate'], df_theory['success rate'], label='SDP Bound', color='dodgerblue',
                 linestyle='-')
        ax0.plot(df_theory['fixed rate'], df_theory['error rate'], color='firebrick', linestyle='-')
        ax0.plot(df_theory['fixed rate'], df_theory['failure rate'], color='limegreen', linestyle='-')

        # 4. 선택된 옵티마이저의 에러바 플로팅
        ax0.errorbar(cur_df['fixed rate'], cur_df['success rate'], yerr=cur_df['success std'], fmt='o',
                     color='dodgerblue', capsize=2, label='Success')
        ax0.errorbar(cur_df['fixed rate'], cur_df['error rate'], yerr=cur_df['error std'], fmt='o', color='firebrick',
                     capsize=2, label='Error')
        ax0.errorbar(cur_df['fixed rate'], cur_df['failure rate'], yerr=cur_df['failure std'], fmt='o',
                     color='limegreen', capsize=2, label='Failure')

        # 5. 마우스 호버(Hover) 감지용 투명 Scatter 객체 생성
        p_succ = ax0.scatter(cur_df['fixed rate'], cur_df['success rate'], s=100, picker=5, alpha=0)
        p_err = ax0.scatter(cur_df['fixed rate'], cur_df['error rate'], s=100, picker=5, alpha=0)
        p_fail = ax0.scatter(cur_df['fixed rate'], cur_df['failure rate'], s=100, picker=5, alpha=0)
        state['scatters'] = [p_succ, p_err, p_fail]

        # 6. 하이라이트 빈 마커 재생성
        state['highlight'], = ax0.plot([], [], 'o', markeredgecolor='black', markerfacecolor='none',
                                       markeredgewidth=2, markersize=12, zorder=10, visible=False)

        # 7. 그래프 꾸미기
        ax0.set_xlabel('Fixed Rate')
        ax0.set_ylabel('Probability')
        ax0.set_title(f'VQSD Result [{opt_name}] (Dim={dim}, Overlap={overlap})')
        ax0.legend(loc='upper right', fontsize=10)

        # 8. 옵티마이저가 바뀌었으니 좌측 궤적 그래프도 첫 번째(0) 데이터로 갱신
        state['current_idx'] = None
        update_trajectory(0)
        fig.canvas.draw_idle()

    def update_trajectory(idx):
        cur_df = state['best_df']
        if cur_df is None or len(cur_df) == 0:
            return

        ax1.clear()
        ax2.clear()

        # 현재 옵티마이저 데이터프레임에서 history 로드
        history_raw = cur_df.iloc[idx]['history']
        trajectory_list = list(map(list, zip(*history_raw)))
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
        Q = cur_df.iloc[idx]['fixed rate']
        ax1.set_title(f'Trajectory (Rate = {Q:.3f})')

        ax2.set_ylabel('Lagrangian Value', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        ax2.plot(trajectory[:, -1], label=r'$\mathcal{L}$', color='firebrick', linewidth=1.5)

        fig.canvas.draw_idle()

    def on_hover(event):
        if event.inaxes != ax0:
            if state['highlight'] and state['highlight'].get_visible():
                state['highlight'].set_visible(False)
                state['current_idx'] = None
                fig.canvas.draw_idle()
            return

        cur_df = state['best_df']
        is_hovered = False
        for scatter_obj in state['scatters']:
            cont, ind_dict = scatter_obj.contains(event)
            if cont:
                is_hovered = True
                idx = ind_dict['ind'][0]

                if state['current_idx'] != idx:
                    target_x = cur_df.iloc[idx]['fixed rate']
                    target_y = [cur_df.iloc[idx]['success rate'], cur_df.iloc[idx]['error rate'],
                                cur_df.iloc[idx]['failure rate']]
                    state['highlight'].set_data([target_x] * 3, target_y)
                    state['highlight'].set_visible(True)
                    state['current_idx'] = idx
                    fig.canvas.draw_idle()
                break

        if not is_hovered and state['highlight'].get_visible():
            state['highlight'].set_visible(False)
            state['current_idx'] = None
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax0 and event.button == 1:
            idx = state['current_idx']
            if idx is not None:
                print(f"🖱️ Clicked Fixed Rate index: {idx}, Updating Trajectory...")
                update_trajectory(idx)

    # 이벤트 연결 및 초기화
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    radio.on_clicked(update_main_plot)

    # 첫 번째 옵티마이저로 초기 플롯 렌더링
    update_main_plot(optimizers[0])

    # 레이아웃 깨짐 방지를 위해 ax_radio 위치가 적용된 뒤에 tight_layout 호출 시 주의 (필요 시 제외)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_qsd_results(script='experiment', dim=3, overlap=0.75, noise=0.1)
