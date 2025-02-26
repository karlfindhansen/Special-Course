import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.append('data')

from data_preprocessing import ArcticDataloader

class InputBlock(nn.Module):
    """
    Custom input block for DeepBedMap.

    TODO: Update my own data to match the input block
    FIXME: Update the input block to match my own data

    Takes in BedMachine (X) : size 16x16 
    REMA Ice Surface Elevation (W1) : size 16x16
    MEaSUREs Ice Surface Velocity x and y components (W2) : size 16x16
    Snow Accumulation (W3) : size 16x16

    Each filter kernel is 4x4 in size, with a stride of 2 and padding of 1.
    """

    def __init__(self, out_channels=32):
        super(InputBlock, self).__init__()

        self.conv_on_X = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )
        
        self.conv_on_W1 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )

        self.conv_on_W2 = nn.Conv2d(
            in_channels=2, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )

        self.conv_on_W3 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )

        # Initialize weights with He (Kaiming) initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, w1, w2, w3):
        """Forward computation based on inputs X, W1, W2, and W3"""

        x_ = self.conv_on_X(x)
        w1_ = self.conv_on_W1(w1)
        w2_ = self.conv_on_W2(w2)
        w3_ = self.conv_on_W3(w3)

        # Concatenate along the channel dimension (dim=1 for NCHW format)
        output = torch.cat((x_, w1_, w2_, w3_), dim=1)
        return output

if __name__ == "__main__":
    input_block = InputBlock()
    dataset = ArcticDataloader(
        bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
        arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
        snow_accumulation_path="data/Snow_acc/...",
        true_crops_folder="data/downscaled_true_crops"
    )

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    for i, batch in enumerate(dataloader):
        if batch['lr_bed_elevation'].shape[0] != 32:
            break
        x = batch['lr_bed_elevation']
        w1 = batch['lr_height_icecap']
        w2 = batch['lr_velocity']
        w3 = torch.randn(batch_size, 1, w2.size()[-1], w2.size()[-1])

    output = input_block(x, w1, w2, w3)
    print("Output size: ", output.size())
    print("Output shape:", output.shape)