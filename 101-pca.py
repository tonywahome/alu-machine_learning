#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load the data
lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

# Normalize the data
data_means = np.mean(data, axis=0)
norm_data = data - data_means

# Perform SVD for PCA
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Get the colormap
cmap = plt.cm.plasma

# Plot each data point with color based on its label
scatter = ax.scatter(
    pca_data[:, 0],  # x coordinates (first PCA component)
    pca_data[:, 1],  # y coordinates (second PCA component)
    pca_data[:, 2],  # z coordinates (third PCA component)
    c=labels,        # color points according to their label
    cmap=cmap,       # use the plasma colormap
    marker='o',      # use circle markers
    s=50,            # set point size
    alpha=0.8        # slight transparency
)

# Set the axis labels
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

# Set the title
ax.set_title('PCA of Iris Dataset')

# Add a color bar to show the mapping of labels to colors
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
cbar.set_label('Species')
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'])

# Adjust the view angle for better visualization
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()
