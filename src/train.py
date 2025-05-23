import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import shutil
from torch.utils.data import DataLoader, random_split

sys.path.append('comparison')
sys.path.append('data')
sys.path.append('src/Model')

from data_preprocessing import ArcticDataset
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel
from train_disc_gen import train_eval_discriminator, train_eval_generator
from utils import get_imgs, save_generator_inputs_canvas_subplots, save_specified_area, plot_val_rmse, plot_fake_real

def train(
    batch_size=128,
    learning_rate=1e-4,
    num_residual_blocks=12,
    residual_scaling=0.1,
    epochs=250,
):
    # Load dataset
    print("Making training dataset")
    dataset = ArcticDataset()

    print("Making unprecise dataset")

    dataset_for_validation = ArcticDataset(
        true_crops=os.path.join("data", "crops", "unprecise_crops", "projected_crops.csv"),
        bedmachine_crops=os.path.join("data", "crops", "unprecise_crops", "original_crops.csv"),
    )

    print("Making comparison dataset")
    dataset_for_comparison = ArcticDataset(region=True)

    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Number of items in train_dataset: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_loader2_unprecise = DataLoader(dataset_for_validation, batch_size=batch_size, shuffle=False) 
    compare_loader = DataLoader(dataset_for_comparison, batch_size=batch_size, shuffle=False)

    print("Number of batches")
    print(f"Training loader: {len(train_loader)}")
    print(f"Validation loader: {len(val_loader)}")
    print(f"Validation unprecise loader: {len(val_loader2_unprecise)}")
    print(f"Compare loader (Kangerlussuaq): {len(compare_loader)}")

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    generator = GeneratorModel(num_residual_blocks=num_residual_blocks, residual_scaling=residual_scaling).to(device)
    discriminator = DiscriminatorModel().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss functions
    mse_loss = nn.MSELoss()
    best_rmse = float("inf")
    epochs_no_improve = 0

    metrics = {
        "discriminator_loss": [],
        "discriminator_accu" : [],
        "generator_loss": [],
        "generator_psnr": [],
        "generator_ssim": [],
    }

    validation_rmse = []
    compare_rmse = []
    
    print(f"Running on {device}")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for imgs in train_loader:
            # Train discriminator
            d_train_loss, d_train_accu = train_eval_discriminator(
                input_arrays=imgs,
                g_model=generator,
                d_model=discriminator,
                d_optimizer=d_optimizer,
                train=True
            )
            
            metrics['discriminator_loss'].append(d_train_loss)
            metrics['discriminator_accu'].append(d_train_accu)

            # Train generator
            g_loss, g_psnr, g_ssim = train_eval_generator(
                input_arrays=imgs,
                g_model=generator,
                d_model=discriminator,
                g_optimizer=g_optimizer,
                train=True
            )

            metrics['generator_loss'].append(g_loss)
            metrics['generator_psnr'].append(g_psnr)
            metrics['generator_ssim'].append(g_ssim)
            
        generator.eval()
        with torch.no_grad():
            val_rmse = 0
            val_rmse_unprecise = 0
            val_rmse_generate = 0
            for imgs in val_loader:
                lr_imgs, hr_imgs = get_imgs(imgs, device)
                preds = generator(lr_imgs[0], lr_imgs[1],lr_imgs[2],lr_imgs[3], lr_imgs[4])
                val_rmse += torch.sqrt(mse_loss(preds, hr_imgs)).item()
            
            plot_fake_real(fake_imgs=preds, real_imgs = hr_imgs, epoch_nr=epoch)
            
            for imgs in val_loader2_unprecise:
                lr_imgs, hr_imgs = get_imgs(imgs, device)
                preds = generator(lr_imgs[0], lr_imgs[1],lr_imgs[2],lr_imgs[3], lr_imgs[4])
                val_rmse_unprecise += torch.sqrt(mse_loss(preds, hr_imgs)).item()

            for imgs in compare_loader:
                lr_imgs, hr_imgs = get_imgs(imgs, device)
                preds = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3], lr_imgs[4])
                val_rmse_generate += torch.sqrt(mse_loss(preds, hr_imgs)).item()

            val_rmse /= len(val_loader)
            val_rmse_unprecise /= len(val_loader2_unprecise)
            val_rmse_generate /= len(compare_loader)
            validation_rmse.append(float(val_rmse_generate))

            if val_rmse_generate < best_rmse:
                best_rmse = val_rmse_generate
                epochs_no_improve = 0
                torch.save(generator.state_dict(), os.path.join("res", "best_generator.pth"))
                torch.save(discriminator.state_dict(), os.path.join("res", "best_discriminator.pth"))

                shutil.rmtree('figures/specified_area')
                os.mkdir('figures/specified_area')

                for imgs in compare_loader:
                    lr_imgs, hr_imgs = get_imgs(imgs, device)
                    preds = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3], lr_imgs[4])
                    save_specified_area(imgs, preds)
                    plot_fake_real(preds, hr_imgs, epoch, output_dir='comparison/figures/fake_real/')

            else:
                epochs_no_improve += 1
                print(f"Validation RMSE did not improve from {best_rmse:.4f}, early stopping counter: {epochs_no_improve}")

            print(f"Epoch {epoch+1}")
            print(", ".join([f"{key}: {item[-1]:.4f}" for key, item in metrics.items()]))
            print(f"Validation RMSE = {val_rmse:.4f}. Unprecise: {val_rmse_unprecise:.4f}")
            print(f"Validation RMSE for generated images = {val_rmse_generate:.4f}")

            if epochs_no_improve >= 30:
                print("Early stopping!")
                plot_val_rmse(validation_rmse, epoch)
                break

        plot_val_rmse(validation_rmse, epoch)

    return best_rmse

if __name__ == "__main__":
    train()
