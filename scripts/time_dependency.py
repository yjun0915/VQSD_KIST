import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from colorspacious import cspace_convert


df = pd.read_csv('../data/results/time_dependency.csv')
opt_method = 'Nelder-Mead'
fig, ax = plt.subplots()
for i, overlap in enumerate([0.25, 0.5, 0.75]):
    for j, tol in enumerate([0.1, 0.01, 0.001]):
        color = cspace_convert([70, 100*(j/3), 360*(i/3)], "CIELCh", "sRGB1")
        color = np.clip(color, 0, 1)
        filtered_df = df[(df['tol']==tol) & (df['overlap']==overlap) & (df['optimizer']==opt_method)]
        if j == 2:
            plt.plot(filtered_df['dim'], filtered_df['time_sec'], 'o-', label=f'overlap={overlap}, tol={tol}', color=color)
        else:
            plt.plot(filtered_df['dim'], filtered_df['time_sec'], 'o--', label=f'overlap={overlap}, tol={tol}', color=color)
plt.legend()
opt_method = 'COBYLA'
for i, overlap in enumerate([0.25, 0.5, 0.75]):
    for j, tol in enumerate([0.1, 0.01, 0.001]):
        color = cspace_convert([70, 100*(j/3), 360*(i/3)], "CIELCh", "sRGB1")
        color = np.clip(color, 0, 1)
        filtered_df = df[(df['tol']==tol) & (df['overlap']==overlap) & (df['optimizer']==opt_method)]
        if j == 2:
            plt.plot(filtered_df['dim'], filtered_df['time_sec'], 'o-', label=f'overlap={overlap}, tol={tol}', color=color)
        else:
            plt.plot(filtered_df['dim'], filtered_df['time_sec'], 'o--', label=f'overlap={overlap}, tol={tol}', color=color)

patch = patches.Polygon(
    np.array([[2.9, 50], [5.1, 275], [5.1, 325], [2.9, 100]]),
    edgecolor='black',
    linestyle='--',
    fill=False,
    facecolor='lightgray',
)
ax.add_patch(patch)

plt.legend(fontsize=5)
plt.tight_layout()
plt.show()
