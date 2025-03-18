import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

img_path_l = 'figures/specified_area/'
df = pd.read_csv(f'{img_path_l}coordinates.csv')

grid_size = 9

df_sorted = df.sort_values(by=["crop_y1", "crop_x1"]).reset_index(drop=True)

# Select the first 81 images for the 9x9 grid
df_selected = df_sorted.iloc[:81]

# Create the plot with no spacing
fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))  # 9x9 inches for a seamless look

# Remove spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Iterate through selected images and plot them without gaps
for idx, (filename, ax) in enumerate(zip(df_selected["image_filename"], axes.ravel())):
    img_path = os.path.join(img_path_l, filename)  # Image path
    try:
        img = plt.imread(img_path)  # Load image
        ax.imshow(img, cmap="terrain")  # Display image
    except FileNotFoundError:
        ax.imshow(np.zeros((10, 10)), cmap="gray")  # Placeholder for missing images
    
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_frame_on(False)  # Hide borders

# Save and show the seamless grid
#plt.savefig("/mnt/data/seamless_9x9_grid.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
