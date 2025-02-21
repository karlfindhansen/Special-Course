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

# Assuming you have GeneratorModel, DiscriminatorModel, and calculate_discriminator_loss defined elsewhere
# from your_module import GeneratorModel, DiscriminatorModel, calculate_discriminator_loss

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
        X = torch.from_numpy(input_arrays["X"]).float().to(device)
        W1 = torch.from_numpy(input_arrays["W1"]).float().to(device)
        W2 = torch.from_numpy(input_arrays["W2"]).float().to(device)
        W3 = torch.from_numpy(input_arrays["W3"]).float().to(device)
        fake_images = g_model(x=X, w1=W1, w2=W2, w3=W3)
        fake_labels = torch.zeros(fake_images.shape[0], 1).to(device)

    # Real groundtruth images
    real_images = torch.from_numpy(input_arrays["Y"]).float().to(device)
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
    X = torch.from_numpy(input_arrays["X"]).float().to(device)
    W1 = torch.from_numpy(input_arrays["W1"]).float().to(device)
    W2 = torch.from_numpy(input_arrays["W2"]).float().to(device)
    W3 = torch.from_numpy(input_arrays["W3"]).float().to(device)
    fake_images = g_model(x=X, w1=W1, w2=W2, w3=W3)

    # Discriminator believes is real
    with torch.no_grad():
        fake_labels = d_model(x=fake_images).float()

    # Real groundtruth images
    real_images = torch.from_numpy(input_arrays["Y"]).float().to(device)
    real_labels = torch.ones(real_images.shape[0], 1).float().to(device)

    # Comparison
    fake_minus_real_target = torch.ones(real_images.shape[0], 1).int().to(device)
    real_minus_fake_target = torch.zeros(real_images.shape[0], 1).int().to(device)
    x_topo = torch.from_numpy(input_arrays["X"][:, :, 1:-1, 1:-1]).float().to(device)

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
    g_ssim = ssim_loss(y_pred=fake_images, y_true=real_images)

    # Generator learning
    if train is True:
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    return g_loss.item(), g_psnr.item(), g_ssim.item()

