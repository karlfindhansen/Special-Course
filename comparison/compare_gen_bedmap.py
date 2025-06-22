import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import tifffile
from tqdm import tqdm

sys.path.extend([
    'data',
    'src/model',
    'src',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
])

from data_preprocessing import ArcticDataset
from src.Model.GeneratorModel import GeneratorModel

def rolling_std(arr, window_size):
    """Compute rolling standard deviation using uniform filter."""
    mean = uniform_filter1d(arr, size=window_size, mode='nearest')
    mean_sq = uniform_filter1d(arr**2, size=window_size, mode='nearest')
    return np.sqrt(mean_sq - mean**2)


def prepare_data(device, batch):
    """Extract and transfer data to the appropriate device."""
    x = batch["lr_bed_elevation"].to(device)
    w1 = batch["height_icecap"].to(device)
    w2 = batch["velocity"].to(device)
    w3 = batch["mass_balance"].to(device)
    w4 = batch["hillshade"].to(device)
    return x, w1, w2, w3, w4


def extract_transects(bed, gen, row_idx=5, crop_border=6, num_images=8):
    """Extract and crop transects from bed and generated images."""
    rows_bed = []
    rows_gen = []
    for i in range(num_images):
        bed_row = bed[5*12+i][0][row_idx, crop_border:-crop_border].numpy()
        gen_row = gen[5*12+i][0][row_idx, crop_border:-crop_border].cpu().numpy()
        rows_bed.append(bed_row)
        rows_gen.append(gen_row)
    return np.concatenate(rows_bed), np.concatenate(rows_gen)


def plot_transect(elevation_bed, elevation_gen, roughness_bed, roughness_gen, title_suffix=""):
    """Plot elevation and roughness transects."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(elevation_bed, label="Elevation BedMachine")
    axs[0].plot(elevation_gen, label="Elevation Generated", color='orange')
    axs[0].set_ylabel("Elevation (m)")
    axs[0].set_title(f"Elevation along concatenated transect {title_suffix}")
    axs[0].legend()

    axs[1].plot(roughness_bed, label="Roughness BedMachine")
    axs[1].plot(roughness_gen, label="Roughness Generated")
    axs[1].set_ylabel("Roughness (m)")
    axs[1].set_xlabel("Pixel index along transect")
    axs[1].set_title(f"Local surface roughness along transect {title_suffix}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("figures/comparison/transect_comparison_karl.png", dpi=150, bbox_inches='tight')
    plt.show()

def save_images_in_canvas(imgs, preds, save_path="figures/comparison/karl/"):
    """Save generated and ground truth images in canvas format with comparison"""
    os.makedirs(save_path, exist_ok=True)
    
    vmin, vmax = -200, 1000

    preds = preds.squeeze(1).cpu().numpy()
    bed = imgs['hr_bed_elevation'].squeeze(1).numpy()

    grid_size = int(np.sqrt(len(preds)))
    pred_height, pred_width = preds.shape[1], preds.shape[2]
    
    canvas_pred = np.zeros((grid_size * pred_height, grid_size * pred_width))
    canvas_bedmachine = canvas_pred.copy()
    canvas_difference = canvas_pred.copy()
    
    for i in range(len(preds)):
        row = i // grid_size
        col = i % grid_size
        
        y1, y2 = row * pred_height, (row + 1) * pred_height
        x1, x2 = col * pred_width, (col + 1) * pred_width
        
        canvas_pred[y1:y2, x1:x2] = preds[i]
        canvas_bedmachine[y1:y2, x1:x2] = bed[i]
        canvas_difference[y1:y2, x1:x2] = preds[i] - bed[i]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(canvas_pred, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted Elevation")
    axes[0].axis('off')
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Elevation [m]')

    im2 = axes[1].imshow(canvas_bedmachine, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1].set_title("Bedmachine Elevation")
    axes[1].axis('off')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Elevation [m]')

    im3 = axes[2].imshow(canvas_difference, cmap='cividis')
    axes[2].set_title("Difference (Predicted - Bedmachine)")
    axes[2].axis('off')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3, label='Elevation Difference [m]')

    plt.imsave(f'{save_path}pred_grid_karl.png', canvas_pred, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imsave(f'{save_path}pred_grid_bedmachine_karl.png', canvas_bedmachine, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imsave(f'{save_path}pred_minus_bedmachine_karl.png', canvas_difference, cmap='cividis')

    plt.tight_layout()
    plt.imshow(canvas_pred)
    plt.savefig(f'{save_path}comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_imgs(imgs, device):
    lr_imgs = (
        imgs['lr_bed_elevation'].to(device),
        imgs['height_icecap'].to(device),
        imgs['velocity'].to(device),
        imgs['mass_balance'].to(device),
        imgs['hillshade'].to(device)
    )
    hr_imgs = imgs['hr_bed_elevation'].to(device)
    return lr_imgs, hr_imgs

# Helper function to find the bounding box of non-NaN values
def get_non_nan_bbox(arr):
    """
    Finds the bounding box (min_row, max_row, min_col, max_col)
    of non-NaN values in a 2D numpy array.
    """
    rows_with_data = np.any(~np.isnan(arr), axis=1)
    cols_with_data = np.any(~np.isnan(arr), axis=0)

    # Handle cases where there might be no non-NaN data
    if not np.any(rows_with_data) or not np.any(cols_with_data):
        return None, None, None, None

    min_row = np.where(rows_with_data)[0][0]
    max_row = np.where(rows_with_data)[0][-1]
    min_col = np.where(cols_with_data)[0][0]
    max_col = np.where(cols_with_data)[0][-1]

    return min_row, max_row, min_col, max_col

def plot_all_predictions_canvas(generator, dataloader, device, save_path='figures/for_report/all.png', title=None):
    """
    Creates a canvas of all predictions placed in their correct spatial locations.
    
    Args:
        generator: The trained generator model
        dataloader: DataLoader containing the validation data
        device: Device to run predictions on
        save_path: Path to save the output image
        title: Title for the plot
    """
    generator.eval()

    print("Making predictions for all of Greenland...")
    
    all_preds = []
    all_crop_locations = []
    
    max_y_abs = max_x_abs = 0
    min_y_abs = min_x_abs = float('inf')
    
    with torch.no_grad():
        for batch_idx, imgs in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
            lr_imgs, _ = get_imgs(imgs, device) 
            
            preds = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3], lr_imgs[4])
            all_preds.extend(preds.cpu())
            
            y1s = imgs['crops']['Original bed']['y_1'].int()
            y2s = imgs['crops']['Original bed']['y_2'].int()
            x1s = imgs['crops']['Original bed']['x_1'].int()
            x2s = imgs['crops']['Original bed']['x_2'].int()
            
            max_y_abs = max(max_y_abs, y2s.max().item())
            max_x_abs = max(max_x_abs, x2s.max().item())
            min_y_abs = min(min_y_abs, y1s.min().item())
            min_x_abs = min(min_x_abs, x1s.min().item())
            
            all_crop_locations.extend(zip(y1s, y2s, x1s, x2s))
            
    canvas_height_abs = max_y_abs - min_y_abs
    canvas_width_abs = max_x_abs - min_x_abs
    canvas = np.full((canvas_height_abs, canvas_width_abs), np.nan, dtype=np.float32)
    
    hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
    vmin, vmax = hr_bed_machine.min()-200, hr_bed_machine.max()
    
    for pred, (y1, y2, x1, x2) in zip(all_preds, all_crop_locations):
        rel_y1 = y1 - min_y_abs + 12
        rel_y2 = y2 - min_y_abs - 12
        rel_x1 = x1 - min_x_abs + 12 
        rel_x2 = x2 - min_x_abs - 12
        
        pred_np = pred.squeeze().numpy()[12:-12, 12:-12]
        canvas[rel_y1:rel_y2, rel_x1:rel_x2] = pred_np

    min_row, max_row, min_col, max_col = get_non_nan_bbox(canvas)

    if min_row is None: 
        print("No valid data found in the canvas to plot.")
        plt.close()
        return None

    cropped_canvas = canvas[min_row : max_row + 1, min_col : max_col + 1]
    masked_cropped_canvas = np.ma.masked_invalid(cropped_canvas)

    plot_extent = (min_col, max_col + 1, max_row + 1, min_row) 

    fig_width = 20 
    cropped_height, cropped_width = masked_cropped_canvas.shape
    fig_height = fig_width * (cropped_height / cropped_width)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111) 

    im = ax.imshow(masked_cropped_canvas, cmap='terrain', vmin=vmin, vmax=vmax, origin='upper', extent=plot_extent)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    cbar = plt.colorbar(im, cax=cax, label='Elevation [m]')
    cbar.ax.tick_params(labelsize=12)
    
    ax.axis('off')
    fig.tight_layout() 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.close()

    tif_save_path = save_path.replace('.png', '.tif')
    tifffile.imwrite(tif_save_path, cropped_canvas.astype(np.float32), dtype='float32')
    


    return cropped_canvas 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    generator = GeneratorModel(num_residual_blocks=48).to(device)
    generator.load_state_dict(torch.load("res/best_generator.pth", map_location=device))
    generator.eval()
    print("Generator model loaded!")

    dataset = ArcticDataset(region='All')
    print(f"Dataset contains {len(dataset)} crops in the specified region!")

    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    with torch.no_grad():
        batch = next(iter(dataloader))
        x, w1, w2, w3, w4 = prepare_data(device, batch)
        generated = generator(x, w1, w2, w3, w4)
        bed = batch['hr_bed_elevation']

        transect_bed, transect_gen = extract_transects(bed, generated)
        roughness_bed = rolling_std(transect_bed, window_size=5)
        roughness_gen = rolling_std(transect_gen, window_size=5)

        #plot_transect(transect_bed, transect_gen, roughness_bed, roughness_gen)
        #save_images_in_canvas(batch, generated)
        plot_all_predictions_canvas(generator, dataloader, device, save_path="figures/comparison/karl/all_predictions_canvas.png", title="All Predictions Canvas")


if __name__ == '__main__':
    main()
