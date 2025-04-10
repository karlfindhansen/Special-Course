import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from skimage.io import imread

img_dir = 'figures/specified_area/'
csv_path = 'data/crops/true_crops/large_crops/original_crops.csv'
netcdf_path = 'data/inputs/Bedmachine/BedMachineGreenland-v5.nc'
save_path = 'comparison/hopeful_plot_with_same_cmap.png'

image_files = sorted(
    [img for img in os.listdir(img_dir) if img.endswith('.png')],
    key=lambda x: int(re.search(r'\d+', x).group())
)

df = pd.read_csv(csv_path)
y1, y2 = df['y_1'].min(), df['y_2'].max()
x1, x2 = df['x_1'].min(), df['x_2'].max()

hr_bed_machine = xr.open_dataset(netcdf_path)['bed']
cropped_bed_machine = hr_bed_machine[y1:y2, x1:x2]

images = [imread(os.path.join(img_dir, img)) for img in image_files[:121]]

rows, cols = 11, 11

stitched_image_1 = np.vstack([
    np.hstack(images[row * cols:(row + 1) * cols]) for row in range(rows)
])

vmin, vmax = hr_bed_machine.min(), hr_bed_machine.max()

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

im1 = axes[0].imshow(stitched_image_1, cmap='terrain', vmin=vmin, vmax=vmax)
axes[0].axis('off')
axes[0].set_title("Generated image")

im2 = axes[1].imshow(cropped_bed_machine, cmap='terrain', vmin=vmin, vmax=vmax)
axes[1].axis('off')
axes[1].set_title("Bedmachine")

cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Elevation')

plt.savefig(save_path, dpi=300)
plt.show()
