import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.color import rgba2rgb, rgb2gray


img_dir = 'figures/specified_area/'
csv_path = 'data/crops//coordinate_crops/original_crops.csv'
netcdf_path = 'data/inputs/Bedmachine/BedMachineGreenland-v5.nc'
save_path = 'comparison/comparison.png'

image_files = sorted(
    [img for img in os.listdir(img_dir) if img.endswith('.png')],
    key=lambda x: int(re.search(r'\d+', x).group())
)

df = pd.read_csv(csv_path)
y1, y2 = df['y_1'].min(), df['y_2'].max()
x1, x2 = df['x_1'].min(), df['x_2'].max()

hr_bed_machine = xr.open_dataset(netcdf_path)['bed']
cropped_bed_machine = hr_bed_machine[y1:y2, x1:x2]

images = [imread(os.path.join(img_dir, img)) for img in image_files]

num_images = len(images)
rows, cols = int(np.sqrt(num_images)), int(np.sqrt(num_images))

generated_img = np.vstack([
    np.hstack(images[row * cols:(row + 1) * cols]) for row in range(rows)
])


fig, axes = plt.subplots(1, 3, figsize=(30, 10))
gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1,0.05])

# Convert to grayscale if it has 4 channels (RGBA)
if generated_img.shape[-1] == 4:
    generated_img_gray = rgb2gray(rgba2rgb(generated_img))
else:
    generated_img_gray = generated_img

cropped_resized = resize(
    cropped_bed_machine.values, 
    generated_img_gray.shape,
    order=1, 
    preserve_range=True, 
    anti_aliasing=True
)

min_val = np.min(cropped_resized)
max_val = np.max(cropped_resized)
gray_image = (cropped_resized - min_val) / (max_val - min_val) 

difference_img = generated_img_gray - gray_image

vmin, vmax = 0, 1

im1 = axes[0].imshow(generated_img, cmap='terrain', vmin=vmin, vmax=vmax)
axes[0].axis('off')
axes[0].set_title("Resized Generated image")

im2 = axes[1].imshow(cropped_resized, cmap='terrain', vmin=vmin, vmax=vmax)
axes[1].axis('off')
axes[1].set_title("Bedmachine")

diff_vmin = difference_img.min()
diff_vmax = difference_img.max()

im3 = axes[2].imshow(difference_img, cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)
axes[2].axis('off')
axes[2].set_title("Difference (Resized Generated - Bedmachine)")

cax = fig.add_subplot(gs[3])
plt.colorbar(im3, cax=cax, label='Elevation Difference [m]')

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()