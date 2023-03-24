import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from skimage.util import random_noise
import pandas as pd

map_pixel_size = 512
resolution = 24e-6  # 24 micron
map_dim = resolution * map_pixel_size  # map dimension in meters
map_size = (map_dim * 1000) ** 2  # map size in mm^2

print(f"[Info] Map size: {map_size} mm^2")
dataset_volume = 118

dataset_dir = 'dataset'
original_dir = os.path.join(dataset_dir, 'original')
noisy_dir = os.path.join(dataset_dir, 'noisy')
label_csv = os.path.join(dataset_dir, 'label.csv')
df = pd.DataFrame(columns=['Seed Number', 'SNR', 'Grain Number', 'Mean Grain Size'])
# Create the dataset directory and subdirectories if they don't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
if not os.path.exists(original_dir):
    os.makedirs(original_dir)
else:
    original_images = [f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f'There are {len(original_images)} images in the original directory.')
if not os.path.exists(noisy_dir):
    os.makedirs(noisy_dir)
else:
    noisy_images = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f'There are {len(noisy_images)} images in the noisy directory.')
existing_num = 0
# Check if the original and noisy directories are not empty and if the label.csv file exists
if (os.path.exists(original_dir) and os.listdir(original_dir)) and \
   (os.path.exists(noisy_dir) and os.listdir(noisy_dir)) and \
   os.path.exists(label_csv):
    print("Both 'original' and 'noisy' directories are not empty, and 'label.csv' exists.")
    df = pd.read_csv(label_csv)
    existing_num = len(noisy_images)
else:
    print("One or more conditions are not met.")
    existing_num = 0



for idx in range(dataset_volume):
    n = np.random.randint(150, 2000)
    SNR = np.random.randint(20, 40)
    
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
    for j, region in enumerate(vor.regions):
        if not -1 in region:
            polygon = [vor.vertices[k] for k in region]
            polygons.append(polygon)
            ax.fill(*zip(*polygon), color=cmap(values[j-1]))

    Trimming_Upper_Limit = 0.8
    Trimming_Lower_limit = 0.2
    ax.set_xlim([Trimming_Lower_limit, Trimming_Upper_Limit])
    ax.set_ylim([Trimming_Lower_limit, Trimming_Upper_Limit])
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    Remaining_Centroid_Mask = (points[:, 0] >= 0.2) & (points[:, 0] <= 0.8) & (points[:, 1] >= 0.2) & (points[:, 1] <= 0.8)
    number_of_centroids = np.count_nonzero(Remaining_Centroid_Mask)
    

    Centroid_X = points[:, 0][Remaining_Centroid_Mask]
    Centroid_Y = points[:, 1][Remaining_Centroid_Mask]
    new_points = np.column_stack((Centroid_X, Centroid_Y))
    # Turn off the axis
    ax.set_axis_off()
    # Show the plot

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    fig.savefig(f"dataset/original/{existing_num+idx+1}.png", dpi=map_pixel_size/10, bbox_inches='tight', pad_inches=0, format='png', transparent=True, facecolor='none')


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

    mean_grain_size = round(np.mean(areas),4)
 
    image = plt.imread(f'dataset/original/{existing_num+idx+1}.png')[:, :, :3]  # keep only the first three channels
    # Add Gaussian noise to the image

    signal = np.mean(image)
    std = signal / (10**(SNR/20))
    noisy_image = random_noise(image, mode='gaussian', var=std**2)

    # Calculate the SNR
    noise = np.std(noisy_image - image)
    snr = round(20 * np.log10(signal / noise),2)
    # print(f"[Info] SNR: {snr} dB")

    # Save the noisy image
    plt.imsave(f"dataset/noisy/{existing_num+idx+1}.png", noisy_image)

    print(f"[Info] image {existing_num+idx+1} with {n} seeds, SNR: {snr} dB, Grain Number: {number_of_centroids}, mean grain size : {mean_grain_size} mm^2 Generated")

    data_entry = {'Seed Number': int(n), 'SNR': snr, 'Grain Number': int(number_of_centroids), 'Mean Grain Size': mean_grain_size}
    data_entry_df = pd.DataFrame(data_entry, index=[0])
    df = pd.concat([df, data_entry_df], ignore_index=True)
    df.to_csv('dataset/label.csv', index=False)

# Save the DataFrame to a CSV file
