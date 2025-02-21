import torch
import torch.nn.functional as F
import numpy as np
import typing
from src.Model.discriminator_loss import calculate_discriminator_loss
from src.Model.GeneratorModel import GeneratorModel
from src.Model.DiscriminatorModel import DiscriminatorModel
from src.Model.ssim_loss import ssim_loss
from src.Model.pnsr import psnr
from src.Model.generator_loss import calculate_generator_loss
from src.Training.train_disc_gen import train_eval_discriminator, train_eval_generator

# Assuming you have train_eval_discriminator and train_eval_generator defined elsewhere
# from your_module import train_eval_discriminator, train_eval_generator

def trainer(
    i: int,  # current epoch
    columns: typing.List[str],  # dataframe column names, i.e. the metric names
    train_loader: torch.utils.data.DataLoader,
    dev_loader: torch.utils.data.DataLoader,
    g_model: torch.nn.Module,  # generator_model
    g_optimizer: torch.optim.Optimizer,  # generator_optimizer
    d_model: torch.nn.Module,  # discriminator_model
    d_optimizer: torch.optim.Optimizer,  # discriminator_optimizer
) -> typing.Dict[str, typing.List[float]]:
    """
    Trains the Super Resolution Generative Adversarial Networks (SRGAN)'s
    Discriminator and Generator components one after another for one epoch.
    Also does evaluation on a development dataset and reports metrics.
    """

    metrics_dict = {mn: [] for mn in columns}  # reset metrics dictionary

    ## Part 1 - Training on training dataset
    g_model.train()
    d_model.train()
    for train_batch in train_loader:
        train_arrays = {key: value.numpy() for key, value in train_batch.items()} #convert to numpy for train_eval functions.
        ## 1.1 - Train Discriminator
        d_train_loss, d_train_accu = train_eval_discriminator(
            input_arrays=train_arrays,
            g_model=g_model,
            d_model=d_model,
            d_optimizer=d_optimizer,
        )
        metrics_dict["discriminator_loss"].append(d_train_loss)
        metrics_dict["discriminator_accu"].append(d_train_accu)

        ## 1.2 - Train Generator
        g_train_loss, g_train_psnr, g_train_ssim = train_eval_generator(
            input_arrays=train_arrays,
            g_model=g_model,
            d_model=d_model,
            g_optimizer=g_optimizer,
        )
        metrics_dict["generator_loss"].append(g_train_loss)
        metrics_dict["generator_psnr"].append(g_train_psnr)
        metrics_dict["generator_ssim"].append(g_train_ssim)

    ## Part 2 - Evaluation on development dataset
    g_model.eval()
    d_model.eval()
    with torch.no_grad():
        for dev_batch in dev_loader:
            dev_arrays = {key: value.numpy() for key, value in dev_batch.items()} #convert to numpy for train_eval functions.
            ## 2.1 - Evaluate Discriminator
            d_dev_loss, d_dev_accu = train_eval_discriminator(
                input_arrays=dev_arrays, g_model=g_model, d_model=d_model, train=False
            )
            metrics_dict["val_discriminator_loss"].append(d_dev_loss)
            metrics_dict["val_discriminator_accu"].append(d_dev_accu)

            ## 2.2 - Evaluate Generator
            g_dev_loss, g_dev_psnr, g_dev_ssim = train_eval_generator(
                input_arrays=dev_arrays, g_model=g_model, d_model=d_model, train=False
            )
            metrics_dict["val_generator_loss"].append(g_dev_loss)
            metrics_dict["val_generator_psnr"].append(g_dev_psnr)
            metrics_dict["val_generator_ssim"].append(g_dev_ssim)

    return metrics_dict
