import sys
import os

sys.path.append('data')
sys.path.append('src/model')
sys.path.append('src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
from data_preprocessing import ArcticDataloader
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import save_image
from src.Model.GeneratorModel import GeneratorModel
import torch
import tqdm
import pandas as pd
#from train import plot_fake_real

region_of_interest_org = pd.read_csv('data/crops/coordinate_crops/original_crops.csv')
region_of_interest_proj = pd.read_csv('data/crops/coordinate_crops/projected_crops.csv')

regions_of_interest = {
    "Projected": {
        'y_1': region_of_interest_proj['y_1'].astype(int).to_list(),
        'x_1': region_of_interest_proj['x_1'].astype(int).to_list(),
        'y_2': region_of_interest_proj['y_2'].astype(int).to_list(),
        'x_2': region_of_interest_proj['x_2'].astype(int).to_list()
    },
    "Original": {
        'y_1': region_of_interest_org['y_1'].astype(int).to_list(),
        'x_1': region_of_interest_org['x_1'].astype(int).to_list(),
        'y_2': region_of_interest_org['y_2'].astype(int).to_list(),
        'x_2': region_of_interest_org['x_2'].astype(int).to_list()
    }
}

if __name__ == '__main__':
    dataset = ArcticDataloader(
        bedmachine_path=os.path.join("data","inputs", "Bedmachine", "BedMachineGreenland-v5.nc"),
        arcticdem_path=os.path.join("data", "inputs", "Surface_elevation", "arcticdem_mosaic_500m_v4.1.tar"),
        ice_velocity_path=os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc"),
        mass_balance_path=os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif"),
        hillshade_path=os.path.join("data", "inputs", "hillshade", "macgregortest_flowalignedhillshade.tif"),
        region=regions_of_interest
    )

    print(regions_of_interest)

    mse_loss = nn.MSELoss()

    print(f"Dataset contains {len(dataset)} crops in the specified region.")
    exit()

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    generator = GeneratorModel().to(device)
    generator.load_state_dict(torch.load("res/best_generator.pth", map_location=device))

    output_dir = "comparison/figures_compare/"

    # Generating images and plotting them
    with torch.no_grad():
        total_mse = 0
        num_samples = 0
        k = 0
        for idx, imgs in enumerate(tqdm.tqdm(dataloader, desc="Generating images")):
            lr_imgs = (
                imgs['lr_bed_elevation'].to(device),
                imgs['height_icecap'].to(device),
                imgs['velocity'].to(device),
                imgs['mass_balance'].to(device),
            )

            generated_imgs = generator(lr_imgs[0], lr_imgs[1], lr_imgs[2], lr_imgs[3])
            real_imgs = imgs['hr_bed_elevation'].to(device)
            
            batch_mse = mse_loss(generated_imgs, real_imgs).item()
            total_mse += batch_mse * real_imgs.size(0)
            num_samples += real_imgs.size(0)
            
            # Plotting all lr_imgs in the batch
            fig, axes = plt.subplots(len(lr_imgs), lr_imgs[0].shape[0], figsize=(15, 5))  # Adjusting for number of images and batch size
            
            for i, lr_img in enumerate(lr_imgs):
                lr_img_np = lr_img.cpu().detach().numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
                
                # If image has 2 channels, plot each channel separately
                if lr_img_np.shape[-1] == 2:  # 2 channels case
                    for j in range(lr_img_np.shape[0]-1):  # Iterate through the batch dimension
                        # Plot each channel of the image
                        axes[i, j].imshow(lr_img_np[j, :, :, 0])  # First channel (grayscale)
                        axes[i, j].set_title(f"C1- B {j + 1}")
                        axes[i, j].axis('off')

                        axes[i, j + 1].imshow(lr_img_np[j, :, :, 1])  # Second channel (grayscale)
                        axes[i, j + 1].set_title(f"C2 - B {j + 1}")
                        axes[i, j + 1].axis('off')
                else:
                    for j in range(lr_img_np.shape[0]):  # Iterate through the batch dimension
                        axes[i, j].imshow(lr_img_np[j])  # Show the image directly (RGB)
                        axes[i, j].set_title(f"IM{i + 1} - B {j + 1}")
                        axes[i, j].axis('off')

            plt.tight_layout()
            #plt.show()
            plt.close()
            #exit()

            # Save the generated images
            for i in range(len(generated_imgs)):
                k += 1

                #plot_fake_real(generated_imgs, real_imgs, k, output_dir, show=False)
                print(generated_imgs[0].squeeze(0).cpu().numpy())
                plt.imshow(generated_imgs[0].squeeze(0).cpu().numpy(), cmap='terrain')
                plt.show()
                exit()

            val_rmse = torch.sqrt(torch.tensor(total_mse / num_samples)).item()
            print(f"Validation RMSE: {val_rmse}")
            

    print(f"Generated images saved to {output_dir}.")