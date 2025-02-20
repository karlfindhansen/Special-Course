import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append('data')

from data_pipeline import cropped_data  

print('os.listdir(..): ', os.listdir('..'))

class DeepbedmapInputBlock(nn.Module):
    """
    Custom input block for DeepBedMap.

    TODO: Update my own data to match the input block
    FIXME: Update the input block to match my own data

    Takes in BedMachine (X) : size 100x100 
    REMA Ice Surface Elevation (W1) : size 100x100
    MEaSUREs Ice Surface Velocity x and y components (W2) : size 100x100
    Snow Accumulation (W3) : size 100x100

    Each filter kernel is 3km by 3km in size, with a 1km stride and no padding.
    """

    def __init__(self, out_channels=32):
        super(DeepbedmapInputBlock, self).__init__()

        self.conv_on_X = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )
        
        self.conv_on_W1 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=(30, 30), stride=(10, 10), padding=(0, 0)
        )

        self.conv_on_W2 = nn.Conv2d(
            in_channels=2, out_channels=out_channels, kernel_size=(6, 6), stride=(2, 2), padding=(0, 0)
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
    #print(cropped_data[3].keys())
    #print(type(cropped_data[3]['height_icecap']))
    #print(cropped_data[3]['height_icecap'].size())
   # exit()

    input_block = DeepbedmapInputBlock()
    batch_size = 128

    for batch in cropped_data:
        x = batch['ice_velocity_x'].unsqueeze(0)
        y = batch['ice_velocity_y'].unsqueeze(0)
        print(x.size())
        xy = torch.concat([x,y])
        print(xy.size())

    x = torch.randn(batch_size,1,11,11) 
    w1 = torch.randn(batch_size,1,110,110)
    w2 = torch.randn(batch_size,2,22,22)
    w3 = torch.randn(batch_size,1,11,11)

    #print(x)
    #exit()

    output = input_block(x, w1, w2, w3)
    print("Output shape:", output.shape)