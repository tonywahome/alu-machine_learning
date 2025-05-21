#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

# Generate data for a mountain elevation
x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create the scatter plot with elevation data
# Use a colormap that represents elevation well (terrain colors)
scatter = ax.scatter(x, y, c=z, cmap='GnBu', alpha=0.8, s=15)

# Set labels and title
ax.set_xlabel('x coordinate (m)')
ax.set_ylabel('y coordinate (m)')
ax.set_title('Mountain Elevation')

# Add a grid for better readability
ax.grid(alpha=0.3)

# Add colorbar and label it
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('elevation (m)')

# Make the plot visually balanced
ax.set_aspect('equal')

plt.tight_layout()
plt.show()