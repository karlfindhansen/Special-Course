import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import xarray as xr
import re
from skimage.io import imread

img_path_l = 'figures/specified_area/'

# Sort filenames numerically based on the numeric part of the filename
images_in_folder = sorted(
    [img for img in os.listdir(img_path_l) if img.endswith('.png')],
    key=lambda x: int(re.search(r'\d+', x).group())
)

img_2_path = 'data/crops/true_crops/large_crops/original_crops.csv'
df = pd.read_csv(img_2_path)
y1,y2 = min(df['y_1']), max(df['y_2'])
x1,x2 = min(df['x_1']), max(df['x_2'])

hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
cropped_bed_machine = hr_bed_machine[y1:y2, x1:x2]

# Read and resize images to ensure they have the same dimensions
images = []
for img_file in images_in_folder[:121]:  # Limit to the first 121 images (11x11 grid)
    img = imread(os.path.join(img_path_l, img_file))
    images.append(img)

# Determine the number of rows and columns
rows, cols = 11, 11

# Stitch images together into a single large image
stitched_image_1 = np.vstack([
    np.hstack(images[row * cols:(row + 1) * cols]) for row in range(rows)
])

# For demonstration, use the same stitched image as the second plot
stitched_image_2 = cropped_bed_machine  # Replace this with another stitched image if needed

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

# Plot the first stitched image
axes[0].imshow(stitched_image_1, cmap='terrain')
axes[0].axis('off')  # Turn off axes
axes[0].set_title("Stitched Image 1")

# Plot the second stitched image
axes[1].imshow(stitched_image_2, cmap='terrain')
axes[1].axis('off')  # Turn off axes
axes[1].set_title("Stitched Image 2")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()