import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from colorspacious import cspace_convert


df = pd.read_csv('../data/results/time_dependency.csv')
opt_method = 'Nelder-Mead'
fig, ax = plt.subplots()
ax_ = ax.twinx()
lines = []
for i, overlap in enumerate([0.75]):
    for j, tol in enumerate([0.01]):
        color = cspace_convert([70, 100*(j/3), 360*(i/3)], "CIELCh", "sRGB1")
        color = np.clip(color, 0, 1)
        filtered_df = df[(df['tol']==tol) & (df['overlap']==overlap) & (df['optimizer']==opt_method)]
        if j == 2:
            ax.plot(filtered_df['dim'], filtered_df['time_sec']/1000, 'o-', label=f'overlap={overlap}, tol={tol}', color=color)
        else:
            ax.plot(filtered_df['dim'], filtered_df['time_sec']/1000, 'o--', label=f'overlap={overlap}, tol={tol}', color=color)
plt.legend()
opt_method = 'COBYLA'
for i, overlap in enumerate([0.75]):
    for j, tol in enumerate([0.01]):
        color = cspace_convert([70, 100*(j/3), 360*(i/3)], "CIELCh", "sRGB1")
        color = np.clip(color, 0, 1)
        filtered_df = df[(df['tol']==tol) & (df['overlap']==overlap) & (df['optimizer']==opt_method)]
        if j == 2:
            ax.plot(filtered_df['dim'], filtered_df['time_sec']/1000, 'o-', color=color)
        else:
            ax.plot(filtered_df['dim'], filtered_df['time_sec']/1000, 'o--', color=color)


opt_method = 'experiment'
for i, overlap in enumerate([0.75]):
    for j, tol in enumerate([0.01]):
        color = cspace_convert([70, 100*(2/3), 360*(2/3)], "CIELCh", "sRGB1")
        color = np.clip(color, 0, 1)
        filtered_df = df[(df['tol']==tol) & (df['overlap']==overlap) & (df['optimizer']==opt_method)]
        ax_.plot(filtered_df['dim'], filtered_df['time_sec']/(12*60), 'o-', color=color)


patch = patches.Polygon(
    np.array([[2.9, 40/1000], [8.1, 630/1000], [8.1, 700/1000], [2.9, 110/1000]]),
    edgecolor='black',
    linestyle='--',
    fill=False,
    facecolor='lightgray',
    label='Nelder-Mead'
)
ax.add_patch(patch)
ax.legend()
ax_.set_ylim([0, 24])

ax.set_xlabel("dim")
ax.set_ylabel("time (sec)")
ax_.set_ylabel("time (min)")

plt.title("simulation(gray) vs. experiment")
# plt.legend(fontsize=7)
plt.tight_layout()
plt.show()
