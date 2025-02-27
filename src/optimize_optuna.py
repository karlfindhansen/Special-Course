import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import sys
import os
import optuna
#import comet_ml
from torch.utils.data import DataLoader, random_split

sys.path.append('data')
sys.path.append('src/Model')

from data_preprocessing import ArcticDataloader
from src.Model.GeneratorModel import GeneratorModel
from src.Model.InputBlock import InputBlock
from src.Model.DiscriminatorModel import DiscriminatorModel


def objective(
    trial: optuna.trial.Trial = optuna.trial.FixedTrial(
        {
            "batch_size_exponent": 7,
            "num_residual_blocks": 12,
            "residual_scaling": 0.1,
            "learning_rate": 1.6e-4,
            "num_epochs": 120,
        }
    ),
    enable_comet_logging: bool = True,
    resume_experiment_key: str = None,
    reload_d_model_weights: bool = False,
) -> float:

    # Start Comet experiment
    #experiment = comet_ml.Experiment(
    #    workspace="weiji14",
    #    project_name="deepbedmap",
    #    disabled=not enable_comet_logging,
    #)

    # Load dataset
    dataset = ArcticDataloader(bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
                               arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
                               ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
                               snow_accumulation_path="data/Snow_acc/...",
                               true_crops="data/downscaled_true_crops"
    )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Number of items in train_dataset: {len(train_dataset)}")

    #batch_size = 2 ** trial.suggest_int(32, 7, 7)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    #learning_rate = trial.suggest_float("learning_rate", 1.0e-4, 2.0e-4, step=0.1e-4)
    #num_residual_blocks = trial.suggest_int("num_residual_blocks", 12, 12)
    #residual_scaling = trial.suggest_float("residual_scaling", 0.1, 0.3, step=0.05)

    learning_rate = 1.0e-4
    num_residual_blocks = 12
    residual_scaling = 0.2

    # Initialize models
    #input_block = InputBlock()
    generator = GeneratorModel(num_residual_blocks=num_residual_blocks, residual_scaling=residual_scaling).cuda()
    discriminator = DiscriminatorModel().cuda()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    #epochs = trial.suggest_int("num_epochs", 15, 150)
    epochs = 30
    best_rmse = float("inf")

    #for batch in train_loader:
        #print(batch)
    #    break
    #exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for imgs in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            lr_imgs = (
                imgs['lr_bed_elevation'].to(device),
                imgs['lr_height_icecap'].to(device),
                imgs['lr_velocity'].to(device),
                imgs['lr_snow_accumulation'].to(device)
            )
            hr_imgs = (
                imgs['hr_bed_elevation'].to(device),
                imgs['hr_height_icecap'].to(device),
                imgs['hr_velocity'].to(device),
                imgs['hr_snow_accumulation'].to(device)
            )

            # Train Discriminator
            fake_imgs = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3]).detach()
            #print(hr_imgs[0].size())
            d_real = discriminator(hr_imgs[0])
            d_fake = discriminator(fake_imgs)

            d_loss = bce_loss(d_real, torch.ones_like(d_real)) + bce_loss(d_fake, torch.zeros_like(d_fake))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            fake_imgs = generator(lr_imgs)
            g_adv_loss = bce_loss(discriminator(fake_imgs), torch.ones_like(d_real))
            g_pixel_loss = mse_loss(fake_imgs, hr_imgs)

            g_loss = g_adv_loss + g_pixel_loss
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # Validate model
        generator.eval()
        with torch.no_grad():
            val_rmse = 0
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.cuda(), hr_imgs.cuda()
                preds = generator(lr_imgs)
                val_rmse += torch.sqrt(mse_loss(preds, hr_imgs)).item()

        val_rmse /= len(val_loader)
        #experiment.log_metric("val_rmse", val_rmse, step=epoch)

        if val_rmse < best_rmse:
            best_rmse = val_rmse

        trial.report(val_rmse, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    #experiment.end()
    return best_rmse

if __name__ == "__main__":
    n_trials = 1
    #if n_trials == 1:  # Run training once only
    objective(optuna.trial.FixedTrial({"enable_livelossplot": True, "enable_comet_logging": False}))
    # elif n_trials > 1:  # Perform hyperparameter tuning
    #     hostname = os.uname().nodename
    #     tpe_seed = len(hostname) + int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))
        
    #     sampler = optuna.samplers.TPESampler(
    #         seed=tpe_seed, **optuna.samplers.TPESampler.hyperopt_parameters()
    #     )
    #     study = optuna.create_study(
    #         study_name="DeepBedMap_tuning",
    #         storage=f"sqlite:///model/logs/train_on_{hostname}.db",
    #         sampler=sampler,
    #         pruner=optuna.pruners.HyperbandPruner(
    #             min_resource=15,
    #             max_resource=150,
    #             reduction_factor=3,
    #         ),
    #         load_if_exists=True,
    #     )
    #     study.optimize(func=objective, n_trials=n_trials, n_jobs=1)