import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

os.chdir(r'D:\pythongraphs\structures')

data = np.genfromtxt("torsion28100cr01.xvg", skip_header=17)
df = pd.DataFrame(data)
df.rename(columns={0: "Time"}, inplace=True)

T_max = 450
T_min = 100
df["Temperature"] = T_max - (350 / (df["Time"].max() - df["Time"].min())) * (df["Time"] - df["Time"].min())

torsion_angles = df.iloc[:, 1:-1].values  
temperatures = df["Temperature"].values

num_bins = 200
bins = np.linspace(-200, 200, num_bins)

probability_distributions = np.array([
    np.histogram(torsion, bins=bins, density=True)[0] for torsion in torsion_angles
])

T_fine = np.linspace(temperatures.min(), temperatures.max(), 300)
B_fine = np.linspace(-200, 200, 300)
T_grid, B_grid = np.meshgrid(T_fine, B_fine)

probability_interp = griddata(
    (np.repeat(temperatures, num_bins - 1), np.tile(bins[:-1], len(temperatures))),
    probability_distributions.flatten(),
    (T_grid, B_grid),
    method="cubic"
)

probability_interp = gaussian_filter(probability_interp, sigma=2)

#plot surface
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(B_grid, T_grid, probability_interp, cmap="rainbow", edgecolor="none", alpha=1.0)

state_angles = [-109, 0, 109]
for angle in state_angles:
    ax.plot([angle, angle], [T_fine.min(), T_fine.max()], 
            [np.nanmin(probability_interp), np.nanmax(probability_interp)], 
            color="red", linestyle="dashed", linewidth=2)

ax.set_xlabel("Torsion Angle (degrees)")
ax.set_ylabel("Temperature (K)")
ax.set_zlabel("P(1/degree)")
cbar_ax = inset_axes(ax, width="3%", height="50%", loc="center left", 
                     bbox_to_anchor=(1.2, 0.0, 1, 1), bbox_transform=ax.transAxes)
cbar = fig.colorbar(surf, cax=cbar_ax)
cbar.set_label("Probability Density")
ax.set_box_aspect([2, 1, 0.5])  # Makes the x-axis twice as broad

plt.show()
