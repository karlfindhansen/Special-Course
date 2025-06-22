import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import pandas as pd
from torch.utils.data import DataLoader
import seaborn as sns
import torch
import sys
from tqdm import tqdm
import rasterio
from scipy.ndimage import uniform_filter1d
from rasterio.transform import from_origin
sys.path.extend(['comparison', 'data', 'src/Model'])
from data_preprocessing import ArcticDataset
from GeneratorModel import GeneratorModel

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

def save_generator_inputs_canvas_subplots(imgs, save_path="figures/specified_area/inputs/"):
    os.makedirs(save_path, exist_ok=True)

    input_names = ["BedMachine", "Arcticdem", "Promice/Velocity", "Mass Balance", "Hillshade"]
    lr_imgs, _ = get_imgs(imgs, device=torch.device("cuda"))

    lr_imgs = [img.cpu().squeeze(1).numpy() for img in lr_imgs]

    for i in len(lr_imgs):
        if input_names[i] == "Promice/Velocity":
            continue

        lr_bed = lr_imgs[i]  
        grid_size = int(np.sqrt(len(lr_bed)))
        img_height, img_width = lr_bed.shape[1], lr_bed.shape[2]
        
        canvas_lr_bed = np.zeros((grid_size * img_height, grid_size * img_width))
        
        for i in range(len(lr_bed)):
            row = i // grid_size
            col = i % grid_size
            y1, y2 = row * img_height, (row + 1) * img_height
            x1, x2 = col * img_width, (col + 1) * img_width
            canvas_lr_bed[y1:y2, x1:x2] = lr_bed[i]

        hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
        vmin, vmax = hr_bed_machine.min(), hr_bed_machine.max()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas_lr_bed, cmap='terrain', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Elevation [m]')
        plt.title(f'{input_names[i]}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_path}{input_names[i]}_grid.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_specified_area(imgs, preds, name):
    hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
    records = []
    save_path="figures/specified_area/"
    csv_filename = f"{save_path}coordinates_{name}.csv"

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(columns=["image_id", "crop_y1", "crop_y2", "crop_x1", "crop_x2", "image_filename"])

    preds = preds.squeeze(1).cpu().numpy()

    klip = 24//2

    grid_size = int(np.sqrt(len(preds)))
    pred_height, pred_width = preds.shape[1]-klip*2, preds.shape[2]-klip*2
    canvas_pred = np.zeros((grid_size * pred_height, grid_size * pred_width))
    canvas_bedmachine = canvas_pred.copy()
    canvas_difference = canvas_pred.copy()
    
    for i in range(len(preds)):

        row = i // grid_size
        col = i % grid_size

        y1, y2 = row * pred_height, (row + 1) * pred_height
        x1, x2 = col * pred_width, (col + 1) * pred_width

        crop_y1, crop_y2 = int(imgs['crops']['Original bed']['y_1'][i]), int(imgs['crops']['Original bed']['y_2'][i])
        crop_x1, crop_x2 = int(imgs['crops']['Original bed']['x_1'][i]), int(imgs['crops']['Original bed']['x_2'][i])

        pred = preds[i][klip:-klip, klip:-klip]
        bed_machine = hr_bed_machine[crop_y1+klip: crop_y2-klip, crop_x1+klip:crop_x2-klip]
        difference = pred - bed_machine
        canvas_pred[y1:y2, x1:x2] = pred
        canvas_bedmachine[y1:y2, x1:x2] = bed_machine
        canvas_difference[y1:y2, x1:x2] = difference

        num_imgs_in_folder = len(os.listdir(save_path))

        records.append({
            "image_id": i,
            "crop_y1": crop_y1, "crop_y2": crop_y2,
            "crop_x1": crop_x1, "crop_x2": crop_x2,
            "image_filename": f'pred_{num_imgs_in_folder+1}.png'
        })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin = min(canvas_pred.min(), canvas_bedmachine.min())
    vmax = max(canvas_pred.max(), canvas_bedmachine.max())

    plt.imsave(f'{save_path}pred_grid_{name}.png', canvas_pred, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imsave(f'{save_path}pred_grid_bedmachine_{name}.png', canvas_bedmachine, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imsave(f'{save_path}pred_minus_bedmachine_{name}.png', canvas_difference, cmap='cividis')

    # Plot 1: Predicted Image Grid
    im1 = axes[0].imshow(canvas_pred, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted Elevation")
    axes[0].axis('off')
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Elevation [m]')

    # Plot 2: Bedmachine Data Grid
    im2 = axes[1].imshow(canvas_bedmachine, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1].set_title("BedMachine Elevation")
    axes[1].axis('off')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Elevation [m]')

    # Plot 3: Difference Grid
    im3 = axes[2].imshow(canvas_difference, cmap='cividis')
    axes[2].set_title("Difference (Predicted - BedMachine)")
    axes[2].axis('off')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3, label='Elevation Difference [m]')

    plt.tight_layout()
    plt.savefig(f'{save_path}comparison_grid_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    new_df = pd.DataFrame(records)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_filename, index=False)

def plot_val_rmse(val_rmse_ls, epochs):
    sns.set_style("darkgrid")  

    x = np.arange(1, epochs + 2)
    y = np.array(val_rmse_ls)

    if epochs > 3:  
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
    else:
        x_smooth, y_smooth = x, y  

    plt.figure(figsize=(8, 5))
    plt.plot(x_smooth, y_smooth, label="Validation RMSE", color="royalblue", linewidth=2.5)

    plt.scatter(x, y, color="red", label="Epoch Points", zorder=3)

    plt.title("Validation RMSE Over Epochs", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=14, labelpad=10)
    plt.ylabel("RMSE", fontsize=14, labelpad=10)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("figures/validation/validation_rmse.png", dpi=300)
    plt.close()

def plot_fake_real(fake_imgs, real_imgs, epoch_nr, output_dir='figures/generated_imgs/', show=False):
    hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
    vmin = hr_bed_machine.min()
    vmax = hr_bed_machine.max()
    sns.set_style("darkgrid")  
    
    fake_imgs = fake_imgs[:4].squeeze(1).cpu().numpy() 
    real_imgs = real_imgs[:4].squeeze(1).cpu().numpy() 

    fig, axes = plt.subplots(4, 2, figsize=(8, 16)) 
    
    for i in range(4):
        axes[i, 0].imshow(fake_imgs[i], cmap="terrain", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Generated (Fake) Image {i+1}", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(real_imgs[i], cmap="terrain", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Ground Truth (Real) Image {i+1}", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

    fig.suptitle(f"Comparison of Fake vs. Real Images (Epoch {epoch_nr})", 
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    plt.savefig(f"{output_dir}fake_real_epoch_{epoch_nr}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    

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


def plot_all_predictions_canvas(generator, dataloader, device, save_path='figures/for_report/all.png', save_tif=False):
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
    vmin, vmax = -800, hr_bed_machine.max()
    
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
    
    return canvas 


def rolling_std(arr, window_size):
    """Compute rolling standard deviation using uniform filter."""
    mean = uniform_filter1d(arr, size=window_size, mode='nearest')
    mean_sq = uniform_filter1d(arr**2, size=window_size, mode='nearest')
    return np.sqrt(mean_sq - mean**2)

def extract_transects(bed, gen, row_idx=18, crop_border=12, num_images=12):
    """Extract and crop transects from bed and generated images."""
    rows_bed = []
    rows_gen = []
    for i in range(num_images):
        bed_row = bed[4*12+i][0][row_idx, crop_border:-crop_border].cpu().numpy()
        gen_row = gen[4*12+i][0][row_idx, crop_border:-crop_border].cpu().numpy()
        rows_bed.append(bed_row)
        rows_gen.append(gen_row)
    return np.concatenate(rows_bed), np.concatenate(rows_gen)


def plot_transect(elevation_bed, elevation_gen, roughness_bed, roughness_gen, save_suffix=""):
    """Plot elevation and roughness transects."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(elevation_bed, label="Elevation BedMachine")
    axs[0].plot(elevation_gen, label="Elevation Generated", color='orange')
    axs[0].set_ylabel("Elevation (m)")
    axs[0].set_title(f"Elevation along concatenated transect {save_suffix}")
    axs[0].legend()

    axs[1].plot(roughness_bed, label="Roughness BedMachine")
    axs[1].plot(roughness_gen, label="Roughness Generated")
    axs[1].set_ylabel("Roughness (m)")
    axs[1].set_xlabel("Pixel index along transect")
    axs[1].set_title(f"Local surface roughness along transect {save_suffix}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/comparison/transect_{save_suffix}.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    region = 'All'
    region_name = 'Kangerlussuaq' if isinstance(region, bool) else 'Middle Region'
    dataset = ArcticDataset(region=region)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    generator = GeneratorModel(num_residual_blocks=48, residual_scaling=0.1).to(device)
    generator.load_state_dict(torch.load('res/best_generator.pth', map_location=device))
    generator.eval()

    batch = next(iter(dataloader))
    bed = batch['hr_bed_elevation'].to(device)
    lr_imgs, hr_imgs = get_imgs(batch, device)
    with torch.no_grad():
        generated = generator(*lr_imgs)

    row_idx = 18 if isinstance(region, bool) else 35
    transect_bed, transect_gen = extract_transects(bed, generated, row_idx=row_idx)
    roughness_bed = rolling_std(transect_bed, window_size=5)
    roughness_gen = rolling_std(transect_gen, window_size=5)
    plot_transect(transect_bed, transect_gen, roughness_bed, roughness_gen, save_suffix=region_name)

    plot_all_predictions_canvas(generator, dataloader, device, save_path='figures/all_preds_utils.png')
