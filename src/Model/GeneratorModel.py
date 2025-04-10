import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys
from ResidualBlocks import ResidualDenseBlock, ResInResDenseBlock
from InputBlock import InputBlock

sys.path.append('data')

from data_preprocessing import ArcticDataloader

class GeneratorModel(nn.Module):
    """
    The generator network which is a deconvolutional neural network.
    Converts a low-resolution input into a super-resolution output.

    Glues the input block with several residual blocks and upsampling layers.

    Parameters:
      num_residual_blocks -- how many Residual-in-Residual Dense Blocks to use
      residual_scaling -- scale factor for residuals before adding to parent branch
      out_channels -- integer representing number of output channels/filters/kernels

    Example:
      A convolved input_shape of (9,9,1) passing through b residual blocks with
      a scaling of 4 and out_channels 1 will result in an image of shape (36,36,1)
    """

    def __init__(
        self,
        inblock_class=InputBlock,  # Pass the input block class as an argument
        resblock_class=ResInResDenseBlock,  # Pass the residual block class as an argument
        num_residual_blocks: int = 12,
        residual_scaling: float = 0.1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.num_residual_blocks = num_residual_blocks
        self.residual_scaling = residual_scaling

        # Initial Input and Residual Blocks
        self.input_block = inblock_class()
        self.pre_residual_conv_layer = nn.Conv2d(
            in_channels=160,  # Adjust based on input_block output
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
        )
        self.residual_network = nn.Sequential(
            *[resblock_class(residual_scaling=residual_scaling) for _ in range(num_residual_blocks)]
        )
        self.post_residual_conv_layer = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
        )

        # Upsampling Layers
        self.post_upsample_conv_layer_1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
        )
        self.post_upsample_conv_layer_2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
        )

        # Final post-upsample convolution layers
        self.final_conv_layer1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
        )
        self.final_conv_layer2 = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, w4: torch.Tensor):
        """
        Forward computation, i.e., evaluate based on input tensors.

        Each input should be a PyTorch tensor.
        """
        # 0 part: Resize inputs to the right scale using convolution
        # and concatenate all inputs
        a0 = self.input_block(x=x, w1=w1, w2=w2, w3=w3, w4=w4)

        # 1st part: Pre-residual convolution k3n64s1
        a1 = self.pre_residual_conv_layer(a0)
        a1 = F.leaky_relu(a1, negative_slope=0.2)

        # 2nd part: Residual blocks k3n64s1
        a2 = self.residual_network(a1)

        # 3rd part: Post-residual convolution k3n64s1
        a3 = self.post_residual_conv_layer(a2)
        a3 = a1 + a3  # Residual connection

        # 4th part: Upsampling (hardcoded to be 4x, actually 2x run twice)
        # Uses Nearest Neighbour Interpolation followed by Convolution2D k3n64s1
        a4_1 = F.interpolate(a3, scale_factor=2, mode='nearest')
        a4_1 = self.post_upsample_conv_layer_1(a4_1)
        a4_1 = F.leaky_relu(a4_1, negative_slope=0.2)

        a4_2 = F.interpolate(a4_1, scale_factor=2, mode='nearest')
        a4_2 = self.post_upsample_conv_layer_2(a4_2)
        a4_2 = F.leaky_relu(a4_2, negative_slope=0.2)

        # 5th part: Generate high-resolution output k3n64s1 and k3n1s1
        a5_1 = self.final_conv_layer1(a4_2)
        a5_1 = F.leaky_relu(a5_1, negative_slope=0.2)
        a5_2 = self.final_conv_layer2(a5_1)

        return a5_2
    
if __name__ == "__main__":

    generator_model = GeneratorModel(
        inblock_class=InputBlock,
        resblock_class=ResInResDenseBlock,
        num_residual_blocks=12,
        residual_scaling=0.1,
        out_channels=1,
    )
    batch_size = 32

    dataset = ArcticDataloader(
                                bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
                                arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
                                ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
                                mass_balance_path="data/Snow_acc/...",
                                true_crops="data/downscaled_true_crops"
    )


    train_size = int(0.9 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    num_batches = len(dataloader)
    
    batch = next(iter(dataloader))
    x = batch['lr_bed_elevation']
    w1 = batch['lr_height_icecap']
    w2 = batch['lr_velocity']
    w3 = batch['lr_snow_accumulation']

    output = generator_model(x, w1, w2, w3)

    
    