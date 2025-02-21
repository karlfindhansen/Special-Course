import torch
import os
import typing
import torch.onnx

def save_model_weights_and_architecture(
    generator_model: torch.nn.Module,
    discriminator_model: torch.nn.Module,
    save_path: str = "model/weights",
) -> typing.Tuple[str, str, str]:
    """
    Save the trained neural network's parameter weights and architecture.
    Weights are saved as .pth files, and architecture as ONNX.
    """

    os.makedirs(name=save_path, exist_ok=True)

    # Save generator/discriminator model's parameter weights in PyTorch format (.pth)
    generator_model_weights_path: str = os.path.join(
        save_path, "srgan_generator_model_weights.pth"
    )

    torch.save(generator_model.state_dict(), generator_model_weights_path)

    discriminator_model_weights_path: str = os.path.join(
        save_path, "srgan_discriminator_model_weights.pth"
    )
    
    torch.save(discriminator_model.state_dict(), discriminator_model_weights_path)

    # Save generator model's architecture in ONNX format
    model_architecture_path: str = os.path.join(
        save_path, "srgan_generator_model_architecture.onnx"
    )

    #dummy inputs for onnx export.
    dummy_input_x = torch.randn(128, 1, 11, 11)
    dummy_input_w1 = torch.randn(128, 1, 110, 110)
    dummy_input_w2 = torch.randn(128, 2, 22, 22)
    dummy_input_w3 = torch.randn(128, 1, 11, 11)

    torch.onnx.export(
        generator_model,
        (dummy_input_x, dummy_input_w1, dummy_input_w2, dummy_input_w3),
        model_architecture_path,
        export_params=True,
        opset_version=10, #change if needed.
        do_constant_folding=True,
        input_names = ['input_x','input_w1','input_w2','input_w3'],
        output_names = ['output'],
        dynamic_axes={'input_x' : {0 : 'batch_size'},    # variable length axes
                        'input_w1' : {0 : 'batch_size'},
                        'input_w2' : {0 : 'batch_size'},
                        'input_w3' : {0 : 'batch_size'},
                        'output' : {0 : 'batch_size'}})

    return (
        generator_model_weights_path,
        discriminator_model_weights_path,
        model_architecture_path,
    )

