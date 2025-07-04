import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block made up of 5 Conv2D-LeakyReLU layers.
    Final output has a residual scaling factor.
    """

    def __init__(self, in_out_channels=64, inter_channels=32, residual_scaling=0.1):
        super(ResidualDenseBlock, self).__init__()
        self.residual_scaling = residual_scaling

        self.conv_layer1 = nn.Conv2d(in_out_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv2d(in_out_channels + inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv2d(in_out_channels + 2 * inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.conv_layer4 = nn.Conv2d(in_out_channels + 3 * inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.conv_layer5 = nn.Conv2d(in_out_channels + 4 * inter_channels, in_out_channels, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        a0 = x

        a1 = F.leaky_relu(self.conv_layer1(a0), negative_slope=0.2)
        a1_cat = torch.cat((a0, a1), dim=1)

        a2 = F.leaky_relu(self.conv_layer2(a1_cat), negative_slope=0.2)
        a2_cat = torch.cat((a0, a1, a2), dim=1)

        a3 = F.leaky_relu(self.conv_layer3(a2_cat), negative_slope=0.2)
        a3_cat = torch.cat((a0, a1, a2, a3), dim=1)

        a4 = F.leaky_relu(self.conv_layer4(a3_cat), negative_slope=0.2)
        a4_cat = torch.cat((a0, a1, a2, a3, a4), dim=1)

        a5 = self.conv_layer5(a4_cat)

        # Apply residual scaling
        a6 = a5 * self.residual_scaling + a0

        return a6

class ResInResDenseBlock(nn.Module):
    """
    Residual in Residual Dense block made of 3 Residual Dense Blocks

       ------------  ----------  ------------
      |            ||          ||            |
    -----DenseBlock--DenseBlock--DenseBlock-(+)--
      |                                      |
       --------------------------------------

    """

    def __init__(self, denseblock_class=ResidualDenseBlock, residual_scaling=0.1):
        super(ResInResDenseBlock, self).__init__()
        self.residual_scaling = residual_scaling

        self.residual_dense_block1 = denseblock_class(residual_scaling=residual_scaling)
        self.residual_dense_block2 = denseblock_class(residual_scaling=residual_scaling)
        self.residual_dense_block3 = denseblock_class(residual_scaling=residual_scaling)

    def forward(self, x):
        a1 = self.residual_dense_block1(x)
        a2 = self.residual_dense_block2(a1)
        a3 = self.residual_dense_block3(a2)

        # Apply residual scaling
        a4 = a3 * self.residual_scaling + x

        return a4