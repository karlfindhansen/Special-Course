import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import sys
import os
from torch.utils.data import DataLoader, random_split

# For plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

sys.path.append('data')
sys.path.append('src/Model')

from data_preprocessing import ArcticDataloader
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel

def train(
    batch_size=64,
    learning_rate=1.0e-4,
    num_residual_blocks=12,
    residual_scaling=0.2,
    epochs=250,
):
    # Load dataset
    dataset = ArcticDataloader(
        bedmachine_path=os.path.join("data", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "Ice_velocity", "Promice_AVG5year.nc"),
        snow_accumulation_path="data/Snow_acc/...",
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Number of items in train_dataset: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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
                imgs['snow_accumulation'].to(device),
            )
            hr_imgs = imgs['hr_bed_elevation'].to(device)

            fake_imgs = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3]).detach()
            d_real = discriminator(hr_imgs)
            d_fake = discriminator(fake_imgs)
            d_loss = bce_loss(d_real, torch.ones_like(d_real)) + bce_loss(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            fake_imgs = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3])
            g_adv_loss = bce_loss(discriminator(fake_imgs), torch.ones_like(d_real))
            g_pixel_loss = mse_loss(fake_imgs, hr_imgs)
            g_loss = g_adv_loss + g_pixel_loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        generator.eval()
        with torch.no_grad():
            val_rmse = 0
            for imgs in val_loader:
                lr_imgs = (
                    imgs['lr_bed_elevation'].to(device),
                    imgs['height_icecap'].to(device),
                    imgs['velocity'].to(device),
                    imgs['snow_accumulation'].to(device),
                )
                hr_imgs = imgs['hr_bed_elevation'].to(device)
                preds = generator(lr_imgs[0], lr_imgs[1],lr_imgs[2],lr_imgs[3])
                val_rmse += torch.sqrt(mse_loss(preds, hr_imgs)).item()
        
        plot_fake_real(fake_imgs=preds, real_imgs = hr_imgs, epoch_nr=epoch)
        val_rmse /= len(val_loader)
        validation_rmse.append(float(val_rmse))
        print(f"Epoch {epoch+1}: Validation RMSE = {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(generator.state_dict(), os.path.join("res", "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join("res", "best_discriminator.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation RMSE did not improve from {best_rmse:.4f}, early stopping counter: {epochs_no_improve}")

        if epochs_no_improve >= 10:
            print("Early stopping!")
            break

    plot_val_rmse(validation_rmse, epochs)

    return best_rmse

def plot_val_rmse(val_rmse_ls, epochs):
    sns.set_style("darkgrid")  # Modern seaborn style

    # Ensure we have proper x values (epochs)
    x = np.arange(1, epochs + 1)
    y = np.array(val_rmse_ls)

    # Create smooth curve using cubic spline interpolation (only if enough points exist)
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
    sns.set_style("darkgrid")  
    
    # Convert to numpy arrays and remove the batch dimension
    fake_imgs = fake_imgs[:4].squeeze(1).cpu().numpy()  # Take the first 4 fake images
    real_imgs = real_imgs[:4].squeeze(1).cpu().numpy()  # Take the first 4 real images

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    
    for i in range(4):
        # Plot fake images
        axes[i // 2, i % 2].imshow(fake_imgs[i], cmap="terrain")
        axes[i // 2, i % 2].set_title(f"Generated (Fake) Image {i+1}", fontsize=12, fontweight="bold")
        axes[i // 2, i % 2].axis("off")
        
        # Plot real images
        axes[i // 2, (i % 2) + 2].imshow(real_imgs[i], cmap="terrain")
        axes[i // 2, (i % 2) + 2].set_title(f"Ground Truth (Real) Image {i+1}", fontsize=12, fontweight="bold")
        axes[i // 2, (i % 2) + 2].axis("off")

    # Add a super title for the entire plot
    fig.suptitle(f"Comparison of Fake vs. Real Images (Epoch {epoch_nr})", 
                 fontsize=16, fontweight="bold", y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{output_dir}fake_real_epoch_{epoch_nr}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


    

if __name__ == "__main__":
    train()
