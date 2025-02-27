import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import typing

sys.path.append('src/Model')

from discriminator_loss import calculate_discriminator_loss
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel
from InputBlock import InputBlock
from ssim_loss import ssim_loss_func
from pnsr import psnr
from generator_loss import calculate_generator_loss

sys.path.append('data')

from data_preprocessing import ArcticDataloader

def train_eval_discriminator(
    input_arrays: typing.Dict[str, np.ndarray],
    g_model: GeneratorModel,
    d_model: DiscriminatorModel,
    d_optimizer: torch.optim.Optimizer = None,
    train: bool = True,
) -> typing.Tuple[float, float]:
    """
    Trains the Discriminator within a Super Resolution Generative Adversarial Network.
    Discriminator is trainable, Generator is not trained (only produces predictions).
    Steps:
    - Generator produces fake images
    - Fake images combined with real groundtruth images
    - Discriminator trained with these images and their Fake(0)/Real(1) labels
    """
    if train is True:
        assert d_optimizer is not None  # Optimizer required for neural network training

    device = next(d_model.parameters()).device #get the device of the model.

    # Generator produces fake images
    with torch.no_grad():
        X = input_arrays["X"].float().to(device)
        W1 = input_arrays["W1"].float().to(device)
        W2 = input_arrays["W2"].float().to(device)
        W3 = input_arrays["W3"].float().to(device)
        fake_images = g_model(x=X, w1=W1, w2=W2, w3=W3)
        fake_labels = torch.zeros(fake_images.shape[0], 1).to(device)

    # Real groundtruth images
    real_images = input_arrays["Y"].float().to(device)
    real_labels = torch.ones(real_images.shape[0], 1).to(device)

    # Discriminator comparison
    real_labels_pred = d_model(x=real_images)
    fake_labels_pred = d_model(x=fake_images)
    real_minus_fake_target = torch.ones(real_images.shape[0], 1).to(device)
    fake_minus_real_target = torch.zeros(real_images.shape[0], 1).to(device)

    d_loss = calculate_discriminator_loss(
        real_labels_pred=real_labels_pred,
        fake_labels_pred=fake_labels_pred,
        real_minus_fake_target=real_minus_fake_target,
        fake_minus_real_target=fake_minus_real_target,
    )

    predicted_labels = torch.cat([real_labels_pred, fake_labels_pred])
    groundtruth_labels = torch.cat([real_labels, fake_labels])
    d_accu = torch.mean((torch.sigmoid(predicted_labels).round() == groundtruth_labels).float())

    # Discriminator learning
    if train is True:
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

    return d_loss.item(), d_accu.item()


# Assuming you have GeneratorModel, DiscriminatorModel, calculate_generator_loss, psnr, and ssim_loss_func defined elsewhere
# from your_module import GeneratorModel, DiscriminatorModel, calculate_generator_loss, psnr, ssim_loss_func

def train_eval_generator(
    input_arrays: typing.Dict[str, np.ndarray],
    g_model: torch.nn.Module,
    d_model: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer = None,
    train: bool = True,
) -> typing.Tuple[float, float, float]:
    """
    Evaluates and/or trains the Generator for one minibatch
    within a Super Resolution Generative Adversarial Network.
    Discriminator is not trainable, Generator is trained.
    If train is set to False, only forward pass is run, i.e. evaluation/prediction only
    If train is set to True, forward and backward pass are run, i.e. train with backprop
    Steps:
    - Generator produces fake images
    - Fake images compared with real groundtruth images
    - Generator is trained to be more like real image
    """
    if train is True:
        assert g_optimizer is not None  # Optimizer required for neural network training

    device = next(g_model.parameters()).device

    # Generator produces fake images
    X = input_arrays["X"].float().to(device)
    W1 = input_arrays["W1"].float().to(device)
    W2 = input_arrays["W2"].float().to(device)
    W3 = input_arrays["W3"].float().to(device)
    fake_images = g_model(x=X, w1=W1, w2=W2, w3=W3)

    # Discriminator believes is real
    with torch.no_grad():
        fake_labels = d_model(x=fake_images).float()

    # Real groundtruth images
    real_images = input_arrays["Y"].float().to(device)
    real_labels = torch.ones(real_images.shape[0], 1).float().to(device)

    # Comparison
    fake_minus_real_target = torch.ones(real_images.shape[0], 1).int().to(device)
    real_minus_fake_target = torch.zeros(real_images.shape[0], 1).int().to(device)
    x_topo = input_arrays["X"][:, :, 1:-1, 1:-1].float().to(device)

    g_loss = calculate_generator_loss(
        y_pred=fake_images,
        y_true=real_images,
        fake_labels=fake_labels,
        real_labels=real_labels,
        fake_minus_real_target=fake_minus_real_target,
        real_minus_fake_target=real_minus_fake_target,
        x_topo=x_topo,
    )
    g_psnr = psnr(y_pred=fake_images, y_true=real_images)
    g_ssim = ssim_loss_func(y_pred=fake_images, y_true=real_images)

    # Generator learning
    if train is True:
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    return g_loss.item(), g_psnr.item(), g_ssim.item()

if __name__ == '__main__':
    g_model = GeneratorModel()
    d_model = DiscriminatorModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_model.to(device)
    d_model.to(device)

    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=1e-4, betas=(0.9, 0.999))


    batch_size = 32

    dataset = ArcticDataloader(
                                bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
                                arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
                                ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
                                snow_accumulation_path="data/Snow_acc/...",
                                true_crops="data/true_crops/selected_crops.csv"
    )


    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    batch = next(iter(dataloader))
    input_arrays = {
        "X": batch['lr_bed_elevation'], # low resolution real images
        "W1": batch['height_icecap'],
        "W2": batch['velocity'],
        "W3": batch['snow_accumulation'],
        "Y":  batch['hr_bed_elevation'], #real images
    }
        
    d_loss, d_accu = train_eval_discriminator(
        input_arrays=input_arrays,
        g_model=g_model,
        d_model=d_model,
        d_optimizer=d_optimizer,
        train=True
    )
    print(f"Discriminator Loss: {d_loss}, Accuracy: {d_accu}")

    # Train generator
    g_loss, g_psnr, g_ssim = train_eval_generator(
        input_arrays=input_arrays,
        g_model=g_model,
        d_model=d_model,
        g_optimizer=g_optimizer,
        train=True
    )
    print(f"Generator Loss: {g_loss}, PSNR: {g_psnr}, SSIM: {g_ssim}")