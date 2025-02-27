import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import sys
import os
from torch.utils.data import DataLoader, random_split

sys.path.append('data')
sys.path.append('src/Model')

from data_preprocessing import ArcticDataloader
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel

def train(
    batch_size=32,
    learning_rate=1.0e-4,
    num_residual_blocks=12,
    residual_scaling=0.2,
    epochs=30,
):
    # Load dataset
    dataset = ArcticDataloader(
        bedmachine_path=os.path.join("data", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "Ice_velocity", "Promice_AVG5year.nc"),
        snow_accumulation_path="data/Snow_acc/...",
    )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Number of items in train_dataset: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        print(f"Epoch {epoch+1}: Validation RMSE = {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(generator.state_dict(), os.path.join("res", "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join("res", "best_discriminator.pth"))

    return best_rmse

def plot_fake_real(fake_imgs, real_imgs, epoch_nr):
    # Convert tensors to numpy arrays if they are not already
    fake_imgs = fake_imgs.squeeze(0).squeeze(0).cpu().numpy()
    real_imgs = real_imgs.squeeze(0).squeeze(0).cpu().numpy()

    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    #print(fake_imgs.size())
    #print(real_imgs.size())
    axes[0].imshow(fake_imgs, cmap='gray')
    axes[0].set_title(f'Fake Image - Epoch {epoch_nr}')
    axes[0].axis('off')
    
    axes[1].imshow(real_imgs, cmap='gray')
    axes[1].set_title(f'Real Image - Epoch {epoch_nr}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"figures/genereated/imgs/fake_real_epoch_{epoch_nr}.png")
    plt.show()

    

if __name__ == "__main__":
    train()
