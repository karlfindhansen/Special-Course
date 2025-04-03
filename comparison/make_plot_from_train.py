import matplotlib.pyplot as plt
import numpy as np
import os
import re
from skimage.io import imread

img_path_l = 'figures/specified_area/'

# Sort filenames numerically based on the numeric part of the filename
images_in_folder = sorted(
    [img for img in os.listdir(img_path_l) if img.endswith('.png')],
    key=lambda x: int(re.search(r'\d+', x).group())
)

# Read and resize images to ensure they have the same dimensions
images = []
for img_file in images_in_folder[:121]:  # Limit to the first 121 images (11x11 grid)
    img = imread(os.path.join(img_path_l, img_file))
    images.append(img)

# Determine the number of rows and columns
rows, cols = 10, 10

# Stitch images together into a single large image
stitched_image = np.vstack([
    np.hstack(images[row * cols:(row + 1) * cols]) for row in range(rows)
])

# Plot the stitched image
plt.figure(figsize=(20, 20))
plt.imshow(stitched_image)
plt.axis('off')  # Turn off axes
plt.title("Stitched Image")
plt.show()