import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader

sys.path.extend(['comparison', 'data', 'src/Model'])

from data_preprocessing import ArcticDataset
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel
from train_disc_gen import train_eval_discriminator, train_eval_generator
from utils import get_imgs, save_specified_area, plot_val_rmse, plot_fake_real, plot_all_predictions_canvas, rolling_std, extract_transects, plot_transect

def train(batch_size=128, learning_rate=1e-4, num_residual_blocks=48, residual_scaling=0.1, epochs=150):
    print("Loading datasets...")
    dataset = ArcticDataset()
    dataset_for_validation = ArcticDataset(region='All')
    dataset_for_comparison = ArcticDataset(region=True)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_all = DataLoader(dataset_for_validation, batch_size=batch_size*2, shuffle=False, num_workers=2)
    compare_loader = DataLoader(dataset_for_comparison, batch_size=batch_size*2, shuffle=False, num_workers=2)

    print(f"Train/Val/Compare batches: {len(train_loader)}, {len(val_loader_all)}, {len(compare_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    generator = GeneratorModel(num_residual_blocks=num_residual_blocks, residual_scaling=residual_scaling).to(device)
    discriminator = DiscriminatorModel().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    mse_loss = nn.MSELoss()
    best_rmse = float("inf")
    epochs_no_improve = 0

    metrics = {"discriminator_loss": [], "discriminator_accu": [], "generator_loss": [], "generator_psnr": [], "generator_ssim": []}
    validation_rmse = []

    print(f"Training on {device}")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for imgs in train_loader:
            d_loss, d_acc = train_eval_discriminator(imgs, generator, discriminator, d_optimizer, train=True)
            g_loss, g_psnr, g_ssim = train_eval_generator(imgs, generator, discriminator, g_optimizer, train=True)

            metrics['discriminator_loss'].append(d_loss)
            metrics['discriminator_accu'].append(d_acc)
            metrics['generator_loss'].append(g_loss)
            metrics['generator_psnr'].append(g_psnr)
            metrics['generator_ssim'].append(g_ssim)

        generator.eval()
        with torch.no_grad():
            val_rmse_generate = sum(
                torch.sqrt(mse_loss(generator(*get_imgs(imgs, device)[0]), get_imgs(imgs, device)[1])).item()
                for imgs in compare_loader
            ) / len(compare_loader)

            val_rmse_unprecise = sum(
                torch.sqrt(mse_loss(generator(*get_imgs(imgs, device)[0]), get_imgs(imgs, device)[1])).item()
                for imgs in val_loader_all
            ) / len(val_loader_all)

            validation_rmse.append(val_rmse_generate)
            
            print(f"Epoch {epoch+1}: " + ", ".join([f"{key}: {item[-1]:.4f}" for key, item in metrics.items()]))
            print(f"Validation Generated Area RMSE = {val_rmse_generate:.4f}, Unprecise = {val_rmse_unprecise:.4f}")

            if val_rmse_generate < best_rmse:
                if epoch >= 10:
                    plot_all_predictions_canvas(generator, val_loader_all, device, save_tif = False, save_path="figures/all_preds.png")

                best_rmse = val_rmse_generate
                epochs_no_improve = 0
                torch.save(generator.state_dict(), os.path.join("res", "best_generator.pth"))
                torch.save(discriminator.state_dict(), os.path.join("res", "best_discriminator.pth"))

                for imgs in compare_loader:
                    lr_imgs, hr_imgs = get_imgs(imgs, device)
                    preds = generator(*lr_imgs)
                    save_specified_area(imgs, preds, name=dataset_for_comparison.glacier_name)
                    transect_bed, transect_gen = extract_transects(hr_imgs, preds)
                    roughness_bed = rolling_std(transect_bed, window_size=5)
                    roughness_gen = rolling_std(transect_gen, window_size=5)

                    plot_transect(transect_bed, transect_gen, roughness_bed, roughness_gen, save_suffix=dataset_for_comparison.glacier_name)
            else:
                epochs_no_improve += 1
                print(f"Validation RMSE did not improve from {best_rmse:.4f}, counter: {epochs_no_improve}")
    
            if epochs_no_improve >= 20:
                print("Early stopping!")
                break

        plot_val_rmse(validation_rmse, epoch)

    return best_rmse

if __name__ == "__main__":
    train()
