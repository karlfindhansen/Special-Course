import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import pandas as pd
import seaborn as sns
import torch

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

def save_specified_area(imgs, preds):
    hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
    records = []
    save_path="figures/specified_area/"
    csv_filename = f"{save_path}coordinates.csv"

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(columns=["image_id", "crop_y1", "crop_y2", "crop_x1", "crop_x2", "image_filename"])

    preds = preds.squeeze(1).cpu().numpy()

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

        crop_y1, crop_y2 = int(imgs['crops']['Original bed']['y_1'][i]), int(imgs['crops']['Original bed']['y_2'][i])
        crop_x1, crop_x2 = int(imgs['crops']['Original bed']['x_1'][i]), int(imgs['crops']['Original bed']['x_2'][i])

        canvas_pred[y1:y2, x1:x2] = preds[i]
        canvas_bedmachine[y1:y2, x1:x2] = hr_bed_machine[crop_y1: crop_y2, crop_x1:crop_x2]
        canvas_difference[y1:y2,x1:x2] = preds[i] - hr_bed_machine[crop_y1: crop_y2, crop_x1:crop_x2]

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

    plt.imsave(f'{save_path}pred_grid.png', canvas_pred, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imsave(f'{save_path}pred_grid_bedmachine.png', canvas_bedmachine, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.imsave(f'{save_path}pred_minus_bedmachine.png', canvas_difference, cmap='cividis')

    # Plot 1: Predicted Image Grid
    im1 = axes[0].imshow(canvas_pred, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted Elevation")
    axes[0].axis('off')
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Elevation [m]')

    # Plot 2: Bedmachine Data Grid
    im2 = axes[1].imshow(canvas_bedmachine, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1].set_title("Bedmachine Elevation")
    axes[1].axis('off')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Elevation [m]')

    # Plot 3: Difference Grid
    im3 = axes[2].imshow(canvas_difference, cmap='cividis')
    axes[2].set_title("Difference (Predicted - Bedmachine)")
    axes[2].axis('off')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3, label='Elevation Difference [m]')

    plt.tight_layout()
    plt.savefig(f'{save_path}comparison_grid.png', dpi=300, bbox_inches='tight')
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

    