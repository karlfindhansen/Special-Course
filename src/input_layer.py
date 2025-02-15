from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.append('data')
from data_preprocessing import ArcticDEMDataset

class InputModule(nn.Module):
    def __init__(self, in_channels_list, out_channels=32):
        """
        Initializes the Input Module.
        
        Args:
        - in_channels_list: List of input channels for each input source.
        - out_channels: Number of output channels for each processed input.
        """
        super(InputModule, self).__init__()
        print("Convolutional block")
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ) for in_ch in in_channels_list
        ])
        print("Final Convolution")
        self.final_conv = nn.Conv2d(len(in_channels_list) * out_channels, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        """
        Forward pass of the Input Module.
        
        Args:
        - inputs: List of input tensors [BEDMAP2, REMA, MEaSUREs]
        
        Returns:
        - Tensor of shape (batch_size, 64, H, W) -> Pre-residual representation.
        """
        assert len(inputs) == len(self.conv_blocks), "Mismatch between inputs and conv blocks"
        print(len(inputs) == len(self.conv_blocks))
        # Process each input
        processed_inputs = [conv_block(x) for conv_block, x in tqdm(list(zip(self.conv_blocks, inputs)), total=len(inputs))]
        # Concatenate along the channel dimension
        concatenated = torch.cat(processed_inputs, dim=1)

        # Final convolution
        pre_residual = self.final_conv(concatenated)

        return pre_residual

dataset = ArcticDEMDataset(
        bedmachine_path="data/BedMachineGreenland-v5.nc",
        arcticdem_path="data/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/dataverse_files/Promice_AVG5year.nc"
    )

dataset = DataLoader(dataset, batch_size=1, shuffle=False)

data = next(iter(dataset))

arctic_dem = data['arcticdem']  # (1, 5400, 1880)
bedmachine = data['errbed']  # (1, 18346, 10218)
ice_velocity_x = data['ice_velocity_x']  # (1, 5400, 3000)
ice_velocity_y = data['ice_velocity_y']  # (1, 5400, 3000)

target_size = (5400, 3000)

# Resize all tensors to (1, 5400, 3000) using bilinear interpolation
arctic_dem = F.interpolate(arctic_dem.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
bedmachine = F.interpolate(bedmachine.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
ice_velocity_x = F.interpolate(ice_velocity_x.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
ice_velocity_y = F.interpolate(ice_velocity_y.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

# Combine ice velocity components into a single tensor (2 channels)
velocities_x_y = torch.cat([ice_velocity_x, ice_velocity_y], dim=0)  # Shape: (2, 5400, 3000)

print("Resized Shapes:")
print("ArcticDEM2: ", arctic_dem.shape)  # Expected: (1, 5400, 3000)
print("Bedmachine: ", bedmachine.shape)  # Expected: (1, 5400, 3000)
print("Ice Velocity: ", velocities_x_y.shape)  # Expected: (2, 5400, 3000)

model = InputModule(in_channels_list=[1, 1, 2]) 
output = model([arctic_dem.unsqueeze(0), bedmachine.unsqueeze(0), velocities_x_y.unsqueeze(0)])

print("Output Shape:", output.shape)  # Expected: (batch_size, 64, 5400, 3000)

if __name__ == "__main__":
    # Simulated input data
    batch_size, height, width = 1, 110, 110  # Example size

    bedmachine = torch.randn(batch_size, 1, height, width)
    arctic_dem = torch.randn(batch_size, 2, height, width)
    velocities_x_y = torch.randn(batch_size, 1, height, width)
    print("Inputting in to the model")
    model = InputModule(in_channels_list=[1, 2, 1])
    output = model([bedmachine, arctic_dem, velocities_x_y])

    print("Output shape:", output.shape)  # Should be (batch_size, 64, H, W)