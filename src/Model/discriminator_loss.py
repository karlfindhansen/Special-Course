import torch
import torch.nn.functional as F
import numpy as np

def calculate_discriminator_loss(
    real_labels_pred: torch.Tensor,
    fake_labels_pred: torch.Tensor,
    real_minus_fake_target: torch.Tensor,
    fake_minus_real_target: torch.Tensor,
) -> torch.Tensor:
    """
    This function purely calculates the "Adversarial Loss"
    in a Relativistic Average Generative Adversarial Network (RaGAN).
    It forms the basis for training the Discriminator Network,
    but it is also used as part of the Generator Network's loss function.
    See paper by Jolicoeur-Martineau, 2018 at https://arxiv.org/abs/1807.00734
    for the mathematical details of the RaGAN loss function.
    Original Sigmoid_Cross_Entropy formula:
    -(y * np.log(sigmoid(x)) + (1 - y) * np.log(1 - sigmoid(x)))
    Numerically stable formula:
    -(x * (y - (x >= 0)) - np.log1p(np.exp(-np.abs(x))))
    where y = the target difference between real and fake labels (i.e. 1 - 0 = 1)
          x = the calculated difference between real_labels_pred and fake_labels_pred
    """

    # Calculate arithmetic mean of real/fake predicted labels
    real_labels_pred_avg = torch.mean(real_labels_pred)
    fake_labels_pred_avg = torch.mean(fake_labels_pred)

    # Binary Cross-Entropy Loss with Sigmoid
    real_versus_fake_loss = F.binary_cross_entropy_with_logits(
        input=(real_labels_pred - fake_labels_pred_avg), target=real_minus_fake_target.float()
    )  # let predicted labels from real images be more realistic than those from fake
    fake_versus_real_loss = F.binary_cross_entropy_with_logits(
        input=(fake_labels_pred - real_labels_pred_avg), target=fake_minus_real_target.float()
    )  # let predicted labels from fake images be less realistic than those from real

    # Relativistic average Standard GAN's Discriminator Loss
    d_loss = real_versus_fake_loss + fake_versus_real_loss

    return d_loss

# Example Usage:
if __name__ == '__main__':
    real_labels_pred = torch.tensor([[1.1], [-0.5]])
    fake_labels_pred = torch.tensor([[-0.3], [1.0]])
    real_minus_fake_target = torch.tensor([[1], [1]]).float() #important to be float.
    fake_minus_real_target = torch.tensor([[0], [0]]).float() #important to be float.

    loss = calculate_discriminator_loss(real_labels_pred, fake_labels_pred, real_minus_fake_target, fake_minus_real_target)
    print(loss.item())