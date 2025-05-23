import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.append('data')

from data_preprocessing import ArcticDataset

class InputBlock(nn.Module):
    """
    Custom input block for DeepBedMap.

    Takes in BedMachine (X) : size 16x16 
    ArcticDEM Ice Surface Elevation (W1) : size 16x16
    MEaSUREs Ice Surface Velocity x and y components (W2) : size 16x16
    Snow Accumulation (W3) : size 16x16

    Each filter kernel is 3x3 in size, with a stride of 1 and padding of 0.
    """

    def __init__(self, out_channels=32):
        super(InputBlock, self).__init__()

        self.conv_on_X = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=5, stride=1, padding=(0, 0)
        )
        
        self.conv_on_W1 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=25, stride=5, padding=(0, 0)
        )

        self.conv_on_W2 = nn.Conv2d(
            in_channels=2, out_channels=out_channels, kernel_size=5, stride=1, padding=(0, 0)
        )

        self.conv_on_W3 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=5, stride=1, padding=(0, 0)
        )

        self.conv_on_W4 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=5, stride=1, padding=(0, 0)
        )

        # Initialize weights with He (Kaiming) initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, w1, w2, w3, w4):
        """Forward computation based on inputs X, W1, W2, and W3"""

        x_ = self.conv_on_X(x)
        w1_ = self.conv_on_W1(w1)
        w2_ = self.conv_on_W2(w2)
        w3_ = self.conv_on_W3(w3)
        w4_ = self.conv_on_W4(w4)

        # Concatenate along the channel dimension (dim=1 for NCHW format)
        output = torch.cat((x_, w1_, w2_, w3_, w4_), dim=1)
        return output

if __name__ == "__main__":
    input_block = InputBlock()
    dataset = ArcticDataset()

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size 
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    for i, batch in enumerate(dataloader):
        x = batch['lr_bed_elevation']
        w1 = batch['height_icecap']
        w2 = batch['velocity']
        w3 = batch['mass_balance']
        w4 = batch['hillshade']
        break
    
    output = input_block(x, w1, w2, w3, w4)
    # this is the goal: Output shape: torch.Size([32, 128, 9, 9])
    print("Output shape:", output.shape)