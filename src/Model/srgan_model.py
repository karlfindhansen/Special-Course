import torch
import torch.optim as optim
import torch.cuda
import numpy as np
from src.Model.GeneratorModel import GeneratorModel
from src.Model.DiscriminatorModel import DiscriminatorModel

# Assuming you have GeneratorModel and DiscriminatorModel defined elsewhere
# from your_module import GeneratorModel, DiscriminatorModel

def compile_srgan_model(
    num_residual_blocks: int = 12,
    residual_scaling: float = 0.1,
    learning_rate: float = 1.6e-4,
):
    """
    Instantiate our Super Resolution Generative Adversarial Network (SRGAN) model here.
    The Generator and Discriminator neural networks are created,
    and an Adam loss optimization function is linked to the models.
    Returns:
    1) generator_model
    2) generator_optimizer
    3) discriminator_model
    4) discriminator_optimizer
    """

    # Instantiate our Generator and Discriminator Neural Network models
    generator_model = GeneratorModel(
        num_residual_blocks=num_residual_blocks, residual_scaling=residual_scaling
    )
    discriminator_model = DiscriminatorModel()

    # Transfer models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_model.to(device)
    discriminator_model.to(device)

    # Setup optimizer, using Adam
    generator_optimizer = optim.Adam(
        generator_model.parameters(), lr=learning_rate, eps=1e-8
    )
    discriminator_optimizer = optim.Adam(
        discriminator_model.parameters(), lr=learning_rate, eps=1e-8
    )

    return (
        generator_model,
        generator_optimizer,
        discriminator_model,
        discriminator_optimizer,
    )

# Example Usage (assuming GeneratorModel and DiscriminatorModel are defined):
if __name__ == '__main__':
    class GeneratorModel(torch.nn.Module):
        def __init__(self, num_residual_blocks=12, residual_scaling=0.1):
            super().__init__()
            # Example layers, replace with your actual generator architecture
            self.linear = torch.nn.Linear(10, 20)
        def forward(self, x):
            return self.linear(x)

    class DiscriminatorModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Example layers, replace with your actual discriminator architecture
            self.linear = torch.nn.Linear(20, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    generator_model, generator_optimizer, discriminator_model, discriminator_optimizer = compile_srgan_model()

    print("Generator Model:", generator_model)
    print("Generator Optimizer:", generator_optimizer)
    print("Discriminator Model:", discriminator_model)
    print("Discriminator Optimizer:", discriminator_optimizer)