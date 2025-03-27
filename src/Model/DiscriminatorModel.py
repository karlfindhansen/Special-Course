import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.append('src/Model')

from ResidualBlocks import ResidualDenseBlock, ResInResDenseBlock
from InputBlock import InputBlock
from GeneratorModel import GeneratorModel

sys.path.append('data')

from data_preprocessing import ArcticDataloader

class DiscriminatorModel(nn.Module):
    """
    The discriminator network which is a convolutional neural network.
    Takes ONE high-resolution input image and predicts whether it is
    real or fake on a scale of 0 to 1, where 0 is fake and 1 is real.

    Consists of several Conv2D-BatchNorm-LeakyReLU blocks, followed by
    a fully connected linear layer with LeakyReLU activation and a final
    fully connected linear layer with Sigmoid activation.
    """

    def __init__(self):
        super().__init__()

        # Define convolutional layers
        self.conv_layer0 = nn.Conv2d(
            in_channels=1,  # Adjust based on input channels
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=True,  # Only the first Conv2D layer uses bias
        )
        self.conv_layer1 = nn.Conv2d(64, 64, 4, 2, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(128, 128, 4, 2, 1, bias=True)
        self.conv_layer4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv_layer5 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)  # No downsampling
        self.conv_layer6 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.conv_layer7 = nn.Conv2d(256, 512, 4, 2, 1, bias=True)
        self.conv_layer8 = nn.Conv2d(512, 512, 3, 1, 1, bias=True)
        self.conv_layer9 = nn.Conv2d(512, 512, 3, 1, 1, bias=True) # Keep same size

        # Define batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.batch_norm7 = nn.BatchNorm2d(512)
        self.batch_norm8 = nn.BatchNorm2d(512)
        self.batch_norm9 = nn.BatchNorm2d(512)

        # Define fully connected layers
        self.linear_1 = nn.Linear(512*2*2*2*2, 100)  # Adjust input size based on final conv output
        self.linear_2 = nn.Linear(100, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Forward computation, i.e., evaluate based on input tensor.

        Input:
          x -- A PyTorch tensor of shape (batch_size, channels, height, width)
        Output:
          A PyTorch tensor of shape (batch_size, 1)
        """
        # 1st part: Convolutional Block without Batch Normalization k3n64s1
        a0 = self.conv_layer0(x)
        a0 = F.leaky_relu(a0, negative_slope=0.2)

        # 2nd part: Convolutional Blocks with Batch Normalization k3n{64*f}s{1or2}
        a1 = self.conv_layer1(a0)
        a1 = self.batch_norm1(a1)
        a1 = F.leaky_relu(a1, negative_slope=0.2)
        a2 = self.conv_layer2(a1)
        a2 = self.batch_norm2(a2)
        a2 = F.leaky_relu(a2, negative_slope=0.2)
        a3 = self.conv_layer3(a2)
        a3 = self.batch_norm3(a3)
        a3 = F.leaky_relu(a3, negative_slope=0.2)
        a4 = self.conv_layer4(a3)
        a4 = self.batch_norm4(a4)
        a4 = F.leaky_relu(a4, negative_slope=0.2)
        a5 = self.conv_layer5(a4)
        a5 = self.batch_norm5(a5)
        a5 = F.leaky_relu(a5, negative_slope=0.2)
        a6 = self.conv_layer6(a5)
        a6 = self.batch_norm6(a6)
        a6 = F.leaky_relu(a6, negative_slope=0.2)
        a7 = self.conv_layer7(a6)
        a7 = self.batch_norm7(a7)
        a7 = F.leaky_relu(a7, negative_slope=0.2)
        a8 = self.conv_layer8(a7)
        a8 = self.batch_norm8(a8)
        a8 = F.leaky_relu(a8, negative_slope=0.2)
        a9 = self.conv_layer9(a8)
        a9 = self.batch_norm9(a9)
        a9 = F.leaky_relu(a9, negative_slope=0.2)

        # 3rd part: Flatten, Dense (Fully Connected) Layers and Output
        a10 = torch.flatten(a9, start_dim=1)  # Flatten while keeping batch_size
        a10 = self.linear_1(a10)
        a10 = F.leaky_relu(a10, negative_slope=0.2)
        a11 = self.linear_2(a10)

        # No sigmoid activation here, as it is typically included in the loss function
        return a11
    
if __name__ == '__main__':
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
                                snow_acc_path="data/Snow_acc/...",
                                true_crops="data/true_crops"
    )


    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    for i, batch in enumerate(dataloader):
        if batch['bed_elevation'].shape[0] != 32:
            break
        x = batch['bed_elevation']
        w1 = batch['height_icecap']
        w2 = batch['velocity']
        w3 = torch.randn(batch_size,1,11,11)

    output = generator_model(x, w1, w2, w3)
    disc_model = DiscriminatorModel()

    output = disc_model(output)
