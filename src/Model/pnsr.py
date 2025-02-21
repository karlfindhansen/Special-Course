import torch
import numpy as np

def psnr(y_pred: torch.Tensor, y_true: torch.Tensor, data_range=2 ** 32) -> torch.Tensor:
    """
    Peak Signal-Noise Ratio (PSNR) metric, calculated batchwise.
    See https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    Can take in either numpy (CPU) or torch (GPU/CPU) tensors as input.
    Implementation is same as skimage.measure.compare_psnr with data_range=2**32
    """

    # Calculate Mean Squared Error
    mse = torch.mean(torch.square(y_pred - y_true))

    # Calculate Peak Signal-Noise Ratio, setting MAX_I as 2^32, i.e. max for int32
    return 20 * torch.log10(torch.tensor(data_range, dtype=torch.float32) / torch.sqrt(mse))

# Example Usage:
if __name__ == '__main__':
    y_pred = torch.ones(2, 1, 3, 3)
    y_true = torch.full((2, 1, 3, 3), 2.0)

    psnr_value = psnr(y_pred, y_true)
    print(psnr_value.item()) #use .item() to get the numerical value from the tensor.

    # Example with numpy arrays:
    y_pred_np = np.ones((2, 1, 3, 3))
    y_true_np = np.full((2, 1, 3, 3), 2.0)

    y_pred_torch = torch.from_numpy(y_pred_np).float()
    y_true_torch = torch.from_numpy(y_true_np).float()

    psnr_value_np = psnr(y_pred_torch, y_true_torch)
    print(psnr_value_np.item())