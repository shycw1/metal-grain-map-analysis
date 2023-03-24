import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.util import random_noise
import imageio


map_pixel_size = 512
resolution = 24e-6 # 24 micron 
map_dim= resolution*map_pixel_size # map dimension in meters 
map_size = (map_dim * 1000) ** 2 # map size in mm^2

print(f"[Info] Map size: {map_size} mm^2")


n = 100  # Generate some random points 100 -2000
SNR = 30  # signal-to-noise ratio



points = np.random.rand(n, 2)
# Generate the Voronoi diagram
vor = Voronoi(points)
# Assign a random value to each point
values = np.random.rand(n)
# Create the colormap
cmap = plt.get_cmap('jet')
polygons = []
# Plot the Voronoi diagram with colored patches
fig, ax = plt.subplots(figsize=(10, 10))
for i, region in enumerate(vor.regions):
    if not -1 in region:
        polygon = [vor.vertices[j] for j in region]
        polygons.append(polygon)
        ax.fill(*zip(*polygon), color=cmap(values[i-1]))


Trimming_Upper_Limit = 0.8
Trimming_Lower_limit = 0.2
ax.set_xlim([Trimming_Lower_limit, Trimming_Upper_Limit])
ax.set_ylim([Trimming_Lower_limit, Trimming_Upper_Limit])
# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])
Remaining_Centroid_Mask = (points[:,0]>= 0.2) & (points[:,0]<= 0.8) & (points[:,1] >= 0.2) & (points[:,1] <= 0.8)
number_of_centroids = np.count_nonzero(Remaining_Centroid_Mask)
print(f"[Info] Grain Number: {number_of_centroids}")
Centroid_X = points[:,0][Remaining_Centroid_Mask]
Centroid_Y = points[:,1][Remaining_Centroid_Mask]
new_points = np.column_stack((Centroid_X, Centroid_Y))
# Turn off the axis
ax.set_axis_off()
# Show the plot

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

fig.savefig('grain_map.png', dpi=map_pixel_size/10, bbox_inches='tight', pad_inches=0, format='png', transparent=True, facecolor='none')



vor = Voronoi(points)
# initialize empty list to store areas
areas = []

# loop through each Voronoi region and calculate its area
for i, region in enumerate(vor.regions):
        # get the vertices of the polygon
    polygon = [vor.vertices[j] for j in region]
        # calculate the area of the polygon using Shoelace formula
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    areas.append(area)
areas= np.sort(areas)
areas = areas[int(n*0.2):int(n*0.7)]
areas = np.array(areas)/0.6/0.6*map_dim*map_dim*1e6

mean_grain_size = np.mean(areas)
print(f"[INFO] mean grain size is: {mean_grain_size} mm^2")


# Load the saved image
image = plt.imread('grain_map.png')[:, :, :3]  # keep only the first three channels

# Add Gaussian noise to the image

signal = np.mean(image)
std = signal / (10**(SNR/20))
noisy_image = random_noise(image, mode='gaussian', var=std**2)

# Calculate the SNR
noise = np.std(noisy_image - image)
snr = 20 * np.log10(signal / noise)
print(f"[Info] SNR: {snr} dB")

# Save the noisy image
plt.imsave('noisy_grain_map.png', noisy_image)

