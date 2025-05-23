import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from discriminator_loss import calculate_discriminator_loss
from ssim_loss import ssim_loss_func

def calculate_generator_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    fake_labels: torch.Tensor,
    real_labels: torch.Tensor,
    fake_minus_real_target: torch.Tensor,
    real_minus_fake_target: torch.Tensor,
    x_topo: torch.Tensor,
    content_loss_weighting: float = 1e-2,
    adversarial_loss_weighting: float = 2e-2,
    topographic_loss_weighting: float = 2e-3,
    structural_loss_weighting: float = 5.25e-0,
) -> torch.Tensor:
    """
    This function calculates the weighted sum between
    "Content Loss", "Adversarial Loss", "Topographic Loss", and "Structural Loss"
    which forms the basis for training the Generator Network.
    """
    # Content Loss (L1, Mean Absolute Error) between predicted and groundtruth 2D images
    content_loss = F.l1_loss(y_pred, y_true)

    # Adversarial Loss between 1D labels
    adversarial_loss = calculate_discriminator_loss(
        real_labels_pred=real_labels,
        fake_labels_pred=fake_labels,
        real_minus_fake_target=real_minus_fake_target,  # Zeros (0) instead of ones (1)
        fake_minus_real_target=fake_minus_real_target,  # Ones (1) instead of zeros (0)
    )

    # Topographic Loss (L1, Mean Absolute Error) between predicted and low res 2D images
    topographic_loss = F.l1_loss(
        F.avg_pool2d(y_pred, kernel_size=(4, 4)), x_topo
    )

    # Structural Similarity Loss between predicted and groundtruth 2D images
    structural_loss = 1 - ssim_loss_func(y_pred, y_true)

    # Get generator loss
    weighted_content_loss = content_loss_weighting * content_loss
    weighted_adversarial_loss = adversarial_loss_weighting * adversarial_loss
    weighted_topographic_loss = topographic_loss_weighting * topographic_loss
    weighted_structural_loss = structural_loss_weighting * structural_loss

    g_loss = (
        weighted_content_loss
        + weighted_adversarial_loss
        + weighted_topographic_loss
        + weighted_structural_loss
    )

    return g_loss



if __name__ == '__main__':
    batch_size = 2
    y_pred = torch.ones(batch_size, 1, 12, 12)
    y_true = torch.full((batch_size, 1, 12, 12), 10.0)
    fake_labels = torch.tensor([[-1.2], [0.5]])
    real_labels = torch.tensor([[0.5], [-0.8]])
    fake_minus_real_target = torch.tensor([[1], [1]]).int()
    real_minus_fake_target = torch.tensor([[0], [0]]).int()
    x_topo = torch.full((batch_size, 1, 3, 3), 9.0)

    loss = calculate_generator_loss(y_pred, y_true, fake_labels, real_labels, fake_minus_real_target, real_minus_fake_target, x_topo)
    print(loss)