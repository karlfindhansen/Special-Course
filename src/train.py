import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import sys
import os
import xarray as xr
import shutil
from torch.utils.data import DataLoader, random_split

# For plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('comparison')
sys.path.append('data')
sys.path.append('src/Model')

from data_preprocessing import ArcticDataloader
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel

def train(
    batch_size=128,
    learning_rate=0.0006921253577657427,
    num_residual_blocks=12,
    residual_scaling=0.3,
    epochs=250,
):
    # Load dataset
    dataset = ArcticDataloader(
        bedmachine_path=os.path.join("data","inputs", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "inputs", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc"),
        mass_balance_path=os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif"),
        hillshade_path=os.path.join("data", "inputs", "hillshade", "macgregortest_flowalignedhillshade.tif"),
    )

    dataset_for_validation = ArcticDataloader(
        bedmachine_path=os.path.join("data","inputs", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "inputs", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc"),
        mass_balance_path=os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif"),
        hillshade_path=os.path.join("data", "inputs", "hillshade", "macgregortest_flowalignedhillshade.tif"),
        true_crops=os.path.join("data", "crops", "unprecise_crops", "projected_crops.csv"),
        bedmachine_crops=os.path.join("data", "crops", "unprecise_crops", "original_crops.csv"),
    )

    dataset_for_comparison = ArcticDataloader(
        bedmachine_path=os.path.join("data","inputs", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "inputs", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc"),
        mass_balance_path=os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif"),
        hillshade_path=os.path.join("data", "inputs", "hillshade", "macgregortest_flowalignedhillshade.tif"),
        region=True
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Number of items in train_dataset: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loader2_unprecise = DataLoader(dataset_for_validation, batch_size=batch_size, shuffle=False) 
    compare_loader = DataLoader(dataset_for_comparison, batch_size=batch_size, shuffle=False)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    generator = GeneratorModel(num_residual_blocks=num_residual_blocks, residual_scaling=residual_scaling).to(device)
    discriminator = DiscriminatorModel().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    best_rmse = float("inf")
    epochs_no_improve = 0

    validation_rmse = []
    
    print(f"Running on {device}")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for imgs in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr_imgs = (
                imgs['lr_bed_elevation'].to(device),
                imgs['height_icecap'].to(device),
                imgs['velocity'].to(device),
                imgs['snow_acc'].to(device),
                imgs['hillshade'].to(device)
            )
            hr_imgs = imgs['hr_bed_elevation'].to(device)

            # Train discriminator
            fake_imgs = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3], lr_imgs[4]).detach()
            d_real = discriminator(hr_imgs)
            d_fake = discriminator(fake_imgs)
            d_loss = bce_loss(d_real, torch.ones_like(d_real)) + bce_loss(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train optimizer
            fake_imgs = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3], lr_imgs[4])
            g_adv_loss = bce_loss(discriminator(fake_imgs), torch.ones_like(d_real))
            g_pixel_loss = mse_loss(fake_imgs, hr_imgs)
            g_loss = g_adv_loss + g_pixel_loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        generator.eval()
        with torch.no_grad():
            val_rmse = 0
            val_rmse_unprecise = 0
            val_rmse_generate = 0
            for imgs in val_loader:
                lr_imgs = (
                    imgs['lr_bed_elevation'].to(device),
                    imgs['height_icecap'].to(device),
                    imgs['velocity'].to(device),
                    imgs['snow_acc'].to(device),
                    imgs['hillshade'].to(device)
                )
                hr_imgs = imgs['hr_bed_elevation'].to(device)
                preds = generator(lr_imgs[0], lr_imgs[1],lr_imgs[2],lr_imgs[3], lr_imgs[4])
                val_rmse += torch.sqrt(mse_loss(preds, hr_imgs)).item()

            for imgs in val_loader2_unprecise:
                lr_imgs = (
                    imgs['lr_bed_elevation'].to(device),
                    imgs['height_icecap'].to(device),
                    imgs['velocity'].to(device),
                    imgs['snow_acc'].to(device),
                    imgs['hillshade'].to(device)
                )
                hr_imgs = imgs['hr_bed_elevation'].to(device)
                preds = generator(lr_imgs[0], lr_imgs[1],lr_imgs[2],lr_imgs[3], lr_imgs[4])
                val_rmse_unprecise += torch.sqrt(mse_loss(preds, hr_imgs)).item()

        plot_fake_real(fake_imgs=preds, real_imgs = hr_imgs, epoch_nr=epoch)
        val_rmse /= len(val_loader)
        val_rmse_unprecise /= len(val_loader2_unprecise)
        validation_rmse.append(float(val_rmse))
        print(f"Epoch {epoch+1}: Validation RMSE = {val_rmse:.4f}. Unprecise: {val_rmse_unprecise:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(generator.state_dict(), os.path.join("res", "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join("res", "best_discriminator.pth"))
            epochs_no_improve = 0

            if epoch%5 == 0:
                shutil.rmtree('figures/specified_area')
                os.mkdir('figures/specified_area')

                generator.eval()  
                with torch.no_grad():
                    for imgs in compare_loader:
                        lr_imgs = (
                            imgs['lr_bed_elevation'].to(device),
                            imgs['height_icecap'].to(device),
                            imgs['velocity'].to(device),
                            imgs['snow_acc'].to(device),
                            imgs['hillshade'].to(device),
                        )
                        hr_imgs = imgs['hr_bed_elevation'].to(device)
                        preds = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3], lr_imgs[4])
                        val_rmse_generate += torch.sqrt(mse_loss(preds, hr_imgs)).item()

                        save_specified_area(imgs, preds) 
                        plot_fake_real(preds, hr_imgs, epoch, output_dir='comparison/figures/fake_real/')
                    val_rmse_generate /= len(compare_loader)
                    print(f"Epoch {epoch+1}: Validation RMSE for generated images = {val_rmse_generate:.4f}")
        
        else:
            epochs_no_improve += 1
            print(f"Validation RMSE did not improve from {best_rmse:.4f}, early stopping counter: {epochs_no_improve}")

        if epochs_no_improve >= 10:
            print("Early stopping!")
            break

    plot_val_rmse(validation_rmse, epoch)

    return best_rmse

def save_specified_area(imgs, preds):
    hr_bed_machine = xr.open_dataset('data/inputs/Bedmachine/BedMachineGreenland-v5.nc')['bed']
    vmin = hr_bed_machine.min()
    vmax = hr_bed_machine.max()
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

        crop_y1, crop_y2 = int(imgs['crops']['Original']['y_1'][i]), int(imgs['crops']['Original']['y_2'][i])
        crop_x1, crop_x2 = int(imgs['crops']['Original']['x_1'][i]), int(imgs['crops']['Original']['x_2'][i])

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

    

if __name__ == "__main__":
    train()
