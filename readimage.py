import matplotlib.pyplot as plt
import numpy as np

# Read the PNG image
image = plt.imread('noisy_grain_map.png')

# Print the shape of the image array
print(f"Image shape: {image.shape}")
plt.imshow(image)


# Read the PNG image
image = plt.imread('grain_map.png')

# Print the shape of the image array
print(f"Image shape: {image.shape}")
plt.imshow(image)