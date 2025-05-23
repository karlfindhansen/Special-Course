import torch
import torch.nn.functional as F
import numpy as np
from kornia.losses import SSIMLoss


def ssim_loss_func(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Structural Similarity (SSIM) loss/metric, calculated with default window size of 9.
    See https://en.wikipedia.org/wiki/Structural_similarity
    Can take in either numpy (CPU) or torch (GPU/CPU) tensors as input.
    """
    if not y_pred.shape == y_true.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    ssim_loss = SSIMLoss(window_size=window_size,reduction='mean') #you can change window size here.

    return ssim_loss(y_pred, y_true)

# Example Usage:
if __name__ == '__main__':
    y_pred = torch.ones(2, 1, 9, 9)
    y_true = torch.full((2, 1, 9, 9), 2.0)

    ssim_value = ssim_loss_func(y_pred, y_true)
    print(ssim_value.item())

    # Example with numpy arrays:
    y_pred_np = np.ones((2, 1, 9, 9))
    y_true_np = np.full((2, 1, 9, 9), 2.0)

    y_pred_torch = torch.from_numpy(y_pred_np).float()
    y_true_torch = torch.from_numpy(y_true_np).float()

    ssim_value_np = ssim_loss_func(y_pred_torch, y_true_torch)
    print(ssim_value_np.item())