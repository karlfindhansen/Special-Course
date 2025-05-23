import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split
from data_preprocessing import ArcticDataset
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel

def objective(trial):
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    num_residual_blocks = trial.suggest_int("num_residual_blocks", 4, 16, step=2)
    residual_scaling = trial.suggest_uniform("residual_scaling", 0.1, 0.5)
    crop_size = trial.suggest_int("crop_size", 5, 50, step=5)
    epochs = 30  # Fixed number of epochs
    
    # Load dataset
    dataset = ArcticDataset(
        bedmachine_path=os.path.join("data", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "Ice_velocity", "Promice_AVG5year.nc"),
        mass_balance_path="data/Snow_acc/...",
        crop_size=crop_size
    )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorModel(num_residual_blocks=num_residual_blocks, residual_scaling=residual_scaling).to(device)
    discriminator = DiscriminatorModel().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    best_rmse = float("inf")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for imgs in train_loader:
            lr_imgs = (
                imgs['lr_bed_elevation'].to(device),
                imgs['height_icecap'].to(device),
                imgs['velocity'].to(device),
                imgs['snow_accumulation'].to(device),
            )
            hr_imgs = imgs['hr_bed_elevation'].to(device)
            
            # Train Discriminator
            fake_imgs = generator(*lr_imgs).detach()
            d_real = discriminator(hr_imgs)
            d_fake = discriminator(fake_imgs)
            d_loss = bce_loss(d_real, torch.ones_like(d_real)) + bce_loss(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            fake_imgs = generator(*lr_imgs)
            g_adv_loss = bce_loss(discriminator(fake_imgs), torch.ones_like(d_real))
            g_pixel_loss = mse_loss(fake_imgs, hr_imgs)
            g_loss = g_adv_loss + g_pixel_loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
        
        # Validation
        generator.eval()
        val_rmse = 0
        with torch.no_grad():
            for imgs in val_loader:
                lr_imgs = (
                    imgs['lr_bed_elevation'].to(device),
                    imgs['height_icecap'].to(device),
                    imgs['velocity'].to(device),
                    imgs['snow_accumulation'].to(device),
                )
                hr_imgs = imgs['hr_bed_elevation'].to(device)
                preds = generator(*lr_imgs)
                val_rmse += torch.sqrt(mse_loss(preds, hr_imgs)).item()
        
        val_rmse /= len(val_loader)
        best_rmse = min(best_rmse, val_rmse)
    
    return best_rmse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)