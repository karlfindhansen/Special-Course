import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.metrics import mean_squared_error as mse
from skimage.io import imread
from skimage.transform import resize
import random

def calculate_mse(grid_img, hr_img):
    # Resize the high-resolution image to match the dimensions of the grid image
    hr_img_resized = resize(hr_img, grid_img.shape, anti_aliasing=True)
    # Calculate the MSE between the high-resolution image and the grid of images
    return mse(hr_img_resized, grid_img)

def swap_images(df_sorted):
    idx1, idx2 = random.sample(range(len(df_sorted)), 2)
    df_sorted.iloc[idx1], df_sorted.iloc[idx2] = df_sorted.iloc[idx2].copy(), df_sorted.iloc[idx1].copy()
    return df_sorted

def simulated_annealing(df_sorted, hr_img, img_path_l, iterations=1000, temp=10.0, cooling=0.99):
    best_df_sorted = df_sorted.copy()
    best_mse = float('inf')

    for i in tqdm(range(iterations), desc='Creating new grids'):
        new_df_sorted = swap_images(df_sorted.copy())
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
        plt.subplots_adjust(wspace=0, hspace=0)

        for idx, (filename, ax) in enumerate(zip(new_df_sorted["image_filename"], axes.ravel())):
            img_path = os.path.join(img_path_l, filename)  
            try:
                img = plt.imread(img_path)  
                ax.imshow(img, cmap="terrain") 
            except FileNotFoundError:
                ax.imshow(np.zeros((10, 10)), cmap="gray") 
            
            ax.set_xticks([])  
            ax.set_yticks([]) 
            ax.set_frame_on(False)  

        for ax in axes.ravel()[num_images:]:
            ax.axis('off')

        plt.savefig("comparison/temp_grid.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        grid_img = imread("comparison/temp_grid.png")
        new_mse = calculate_mse(grid_img, hr_img)

        if new_mse < best_mse or np.exp((best_mse - new_mse) / temp) > np.random.rand():
            best_df_sorted = new_df_sorted.copy()
            best_mse = new_mse

        temp *= cooling

    return best_df_sorted

img_path_l = 'figures/specified_area/'
df = pd.read_csv(f'{img_path_l}coordinates.csv')

num_images = len(df)
grid_size = int(np.ceil(np.sqrt(num_images))) 

df_sorted = df.sort_values(by=["crop_y1", "crop_x1"]).reset_index(drop=True)

# Load the high-resolution image
hr_img_path = 'comparison/bed_elevation_hr.png'
hr_img = imread(hr_img_path)

# Optimize the order of images in df_sorted
optimized_df_sorted = simulated_annealing(df_sorted, hr_img, img_path_l)

# Create the final grid of images with the optimized order
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
plt.subplots_adjust(wspace=0, hspace=0)

for idx, (filename, ax) in enumerate(zip(optimized_df_sorted["image_filename"], axes.ravel())):
    img_path = os.path.join(img_path_l, filename)  
    try:
        img = plt.imread(img_path)  
        ax.imshow(img, cmap="terrain") 
    except FileNotFoundError:
        ax.imshow(np.zeros((10, 10)), cmap="gray") 
    
    ax.set_xticks([])  
    ax.set_yticks([]) 
    ax.set_frame_on(False)  

for ax in axes.ravel()[num_images:]:
    ax.axis('off')

plt.savefig("comparison/grid_from_train_optimized.png", dpi=300, bbox_inches="tight", pad_inches=0)

# Load the grid of images
grid_img_path = 'comparison/grid_from_train_optimized.png'
grid_img = imread(grid_img_path)

# Calculate the MSE between the high-resolution image and the grid of images
mse_value = calculate_mse(grid_img, hr_img)
print(f"Mean Squared Error (MSE) between the high-resolution image and the grid of images: {mse_value}")

# Display the high-resolution image and the grid of images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the high-resolution image
ax1.imshow(hr_img, cmap='terrain')
ax1.set_title('High-Resolution Image')
ax1.axis('off')

# Display the grid of images
ax2.imshow(grid_img, cmap='terrain')
ax2.set_title('Grid of Images')
ax2.axis('off')

plt.tight_layout()
plt.savefig('new_fig.png')
plt.show()