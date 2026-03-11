import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from pathlib import Path
from colorspacious import cspace_convert

plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'axes.grid': False})


def load_latest_file(directory, pattern="*.csv", n=1):
    """
    n=1 이면 가장 최신 파일, n=2 이면 두 번째로 최신 파일을 불러옵니다.
    """
    files = list(Path(directory).glob(pattern))
    if not files:
        return None

    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if n > len(files) or n < 1:
        print(f"⚠️ 경고: {n}번째 파일을 찾을 수 없습니다. (총 파일 수: {len(files)}개)")
        return None

    selected_file = files[n - 1]
    print(f"📁 불러온 파일 ({n}번째 최신): {selected_file.name}")  # 확인용 출력
    return selected_file


def plot_qsd_results(script, dim, overlap, n=1):
    sim_dir = Path(f"../data/{script}/dim_{dim}")
    theory_path = Path(f"../data/theory/dim_{dim}/ov{overlap}.csv")

    sim_path = load_latest_file(sim_dir, f"*_ov{overlap}.csv", n=n)
    history_path = load_latest_file(sim_dir, f"*_ov{overlap}_history.csv", n=n)

    if not sim_path or not theory_path:
        print("Cannot find data")
        return

    df_theory = pd.read_csv(theory_path)
    df_sim = pd.read_csv(sim_path)

    cols = ['success rate', 'error rate', 'failure rate']
    row_sums = df_sim[cols].sum(axis=1)
    df_sim[cols] = df_sim[cols].div(row_sums, axis=0)

    fig, (ax1, ax0) = plt.subplots(1, 2, figsize=(10.46, 6), gridspec_kw={'width_ratios': [1, 1.81]})
    ax2 = ax1.twinx()

    ax0.plot(df_theory['fixed rate'], df_theory['success rate'], label='SDP Bound (Theory)', color='dodgerblue', linestyle='-')
    ax0.plot(df_theory['fixed rate'], df_theory['error rate'], label='SDP Bound (Theory)', color='firebrick', linestyle='-')
    ax0.plot(df_theory['fixed rate'], df_theory['failure rate'], label='SDP Bound (Theory)', color='limegreen', linestyle='-')

    best_sim = df_sim.loc[df_sim.groupby('fixed rate')['lagrangian'].idxmax()]

    ax0.plot(best_sim['fixed rate'], best_sim['success rate'], 'o', color='dodgerblue', label='VQE Best Result')
    ax0.plot(best_sim['fixed rate'], best_sim['error rate'], 'o', color='firebrick', label='VQE Best Result')
    ax0.plot(best_sim['fixed rate'], best_sim['failure rate'], 'o', color='limegreen', label='VQE Best Result')
    # 마우스 오버 감지를 위해 데이터 포인트를 변수에 저장 (Numpy 배열로 변환하여 안전성 확보)
    x_data = best_sim['fixed rate'].to_numpy()
    y_succ = best_sim['success rate'].to_numpy()
    y_err = best_sim['error rate'].to_numpy()
    y_fail = best_sim['failure rate'].to_numpy()

    p_succ, = ax0.plot(x_data, y_succ, 'o', color='dodgerblue', label='VQE Best Result')
    p_err, = ax0.plot(x_data, y_err, 'o', color='firebrick', label='VQE Best Result')
    p_fail, = ax0.plot(x_data, y_fail, 'o', color='limegreen', label='VQE Best Result')

    ax0.set_xlabel('Fixed Rate')
    ax0.set_ylabel('Probability')
    ax0.set_title(f'Quantum State Discrimination (Dim={dim}, Overlap={overlap})')
    # ax0.legend()
    # ==========================================
    # 🌟 인터랙티브 로직 (Hover & Click)
    # ==========================================

    # History 데이터 미리 로드
    df_hist = pd.read_csv(history_path) if history_path else None

    # 강조용 빈 마커 생성
    highlight, = ax0.plot([], [], 'o', markeredgecolor='black', markerfacecolor='none',
                          markeredgewidth=2, markersize=12, zorder=10, visible=False)

    hover_state = {'current_idx': None}

    # 1. 궤적 업데이트 함수 (클릭 시 실행됨)
    def update_trajectory(idx):
        if df_hist is None:
            return

        # 기존에 그려진 그래프 초기화 (clear를 하면 label 등도 지워지므로 다시 세팅 필요)
        ax1.clear()
        ax2.clear()

        # 💡 주의: 데이터 저장 방식에 따라 df_hist.columns의 인덱스 오프셋이 필요할 수 있습니다.
        # (예: index 컬럼이 저장되어 있다면 idx + 1 을 해야 할 수도 있음)
        try:
            col_name = df_hist.columns[idx]
        except IndexError:
            print("인덱스 범위를 초과했습니다.")
            return

        # 데이터 파싱
        trajectory_list = df_hist[col_name].dropna().apply(eval).tolist()
        if not trajectory_list:
            return

        selected_fixed_rate = trajectory_list[0]
        trajectory = np.array(trajectory_list[1:])

        num_params = trajectory.shape[1] - 1
        colors = [cspace_convert([70, 60, 360 * (theta / num_params)], "CIELCh", "sRGB1") for theta in
                  range(num_params)]
        colors = np.clip(colors, 0, 1)

        # 파라미터 (ax1) 그리기
        for i in range(int(dim * (dim - 1) / 2)):
            ax1.plot(trajectory[:, i], label=rf'$\theta_{i + 1}$', color=colors[i], linewidth=1)
        for i in range(int(dim * (dim - 1) / 2), (dim ** 2) - 1):
            ax1.plot(trajectory[:, i], label=rf'$\varphi_{i - int(dim * (dim - 1) / 2) + 1}$', color=colors[i],
                     linewidth=1)

        ax1.set_xlabel('Optimization Iterations')
        ax1.set_ylabel('Parameter Value (rad)', color='k')
        ax1.set_title(f'Trajectory (fixed rate = {selected_fixed_rate})')

        # 라그랑지안 (ax2) 그리기
        ax2.set_ylabel('Lagrangian Value', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        ax2.plot(trajectory[:, -1], label=r'$\mathcal{L}$', color='firebrick', linewidth=1.5)

        # 화면 리렌더링
        fig.canvas.draw_idle()

    # 2. 마우스 오버 이벤트 (이전과 동일)
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

    # 3. 클릭 이벤트 (호버된 상태에서 클릭 시 update_trajectory 호출)
    def on_click(event):
        if event.inaxes == ax0 and event.button == 1:  # 1은 마우스 좌클릭
            idx = hover_state['current_idx']
            if idx is not None:
                print(f"🖱️ Clicked Fixed Rate index: {idx}, Updating Trajectory...")
                update_trajectory(idx)

    # 프로그램 시작 시 0번째 인덱스의 궤적을 기본으로 띄워둡니다.
    update_trajectory(0)

    # 이벤트 연결
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    # ==========================================

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_qsd_results(script='simulation', dim=4, overlap=0.75, n=1)
