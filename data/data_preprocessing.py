import os
import tarfile
import torch
import numpy as np
import xarray as xr
import rioxarray
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = None

def create_sliding_window_patches(crop_width, crop_height, patch_size, stride=1):
    """
    Creates a list of patch coordinates using a sliding window approach.

    Args:
        crop_width: Width of the crop.
        crop_height: Height of the crop.
        patch_size: Size of the square patch (both width and height).
        stride: Step size for the sliding window.

    Returns:
        A list of tuples, where each tuple represents a patch: (x_start, y_start, x_end, y_end).
    """

    patches = []
    for y_start in range(0, crop_height - patch_size + 1, stride):
        for x_start in range(0, crop_width - patch_size + 1, stride):
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            patches.append((x_start, y_start, x_end, y_end))
    return patches

class ArcticDataloader(Dataset):
    def __init__(self,
                 bedmachine_path,
                 arcticdem_path,
                 ice_velocity_path,
                 snow_accumulation_path,
                 true_crops_folder,
                 patch_size=11,
                 scale_factor=4):
        """ Initializes the dataset by loading and aligning data from NetCDF and GeoTIFF files. """
        self.bedmachine_data = xr.open_dataset(bedmachine_path)
        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)

        self.ice_velocity_data = xr.open_dataset(ice_velocity_path)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

        extracted_path = "data/arcticdem_extracted"
        os.makedirs(extracted_path, exist_ok=True)
        with tarfile.open(arcticdem_path) as tar:
            tar.extractall(path=extracted_path, filter="fully_trusted")

        tif_file = "arcticdem_mosaic_500m_v4.1_dem.tif"
        self.arcticdem_tif_path = os.path.join(extracted_path, tif_file)

        self.height_map_icecap_data = self.read_icecap_height_data()
        self.height_map_icecap_tensor = self.align_to_bedmachine(self.height_map_icecap_data)

        self.ice_velocity_x_tensor = self.align_to_bedmachine(
            self.ice_velocity_data['land_ice_surface_easting_velocity']
        )
        self.ice_velocity_y_tensor = self.align_to_bedmachine(
            self.ice_velocity_data['land_ice_surface_northing_velocity']
        )

        self.bedmachine = torch.tensor(self.bedmachine_data['bed'].values.astype(np.float32)).unsqueeze(0)

        # Load true crop coordinates
        self.true_crops = [np.load(os.path.join(true_crops_folder, file)) for file in os.listdir(true_crops_folder)]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.crop_patches_info = self._generate_crop_patches_info()

        self.downsample_transform = transforms.Compose([
            transforms.Resize((patch_size // scale_factor, patch_size // scale_factor), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize((patch_size, patch_size), interpolation=transforms.InterpolationMode.BICUBIC)  
        ])

    def read_icecap_height_data(self):
        """ Reads ArcticDEM data as a tensor. """
        arcticdem_data = rioxarray.open_rasterio(self.arcticdem_tif_path)
        arcticdem_data.rio.write_crs("EPSG:3413", inplace=True)
        return arcticdem_data

    def align_to_bedmachine(self, data):
        """ Reprojects and aligns data to match the BedMachine grid. """
        aligned = data.rio.reproject_match(self.bedmachine_data)
        return torch.tensor(aligned.values.astype(np.float32))

    def _generate_crop_patches_info(self):
        """Generates a list of tuples containing crop and patch information."""
        crop_patches_info = []
        #print(f"Height map icecap tensor size: {self.height_map_icecap_tensor.size()}")
        for crop_idx, crop in enumerate(self.true_crops):
            x_1, y_1, x_2, y_2 = crop
            #print(f"Crop {crop_idx} coordinates: {x_1}, {y_1}, {x_2}, {y_2}")

            # Ensure the crop coordinates are within the bounds of the tensor
            if x_1 < 0 or y_1 < 0 or x_1 + y_2 > self.height_map_icecap_tensor.shape[2] or y_1 + x_2 > self.height_map_icecap_tensor.shape[1]:
                print(f"Skipping crop {crop_idx} with coordinates ({x_1}, {y_1}, {x_2}, {y_2}) as it is out of bounds")
                continue

            # Skip crops that are smaller than the patch size
            if x_2 < self.patch_size or y_2 < self.patch_size:
                print(f"Skipping crop {crop_idx} with size {x_2}x{y_2} as it is smaller than patch size {self.patch_size}")
                continue

            # Generate patches within the crop using sliding window
            crop_width = x_2 - x_1
            crop_height = y_2 - y_1

            patches = create_sliding_window_patches(crop_width, crop_height, self.patch_size)

            for patch_x_start, patch_y_start, patch_x_end, patch_y_end in patches:
                crop_patches_info.append((crop_idx, x_1 + patch_x_start, y_1 + patch_y_start, x_1 + patch_x_end, y_1 + patch_y_end))

        #print(len(crop_patches_info))
        return crop_patches_info

    def __len__(self):
        """ Returns the number of available data points. """
        return len(self.crop_patches_info)

    def __getitem__(self, idx):
        """ Returns a fixed-size patch from the crop. """
        crop_idx, patch_x1, patch_y1, patch_x2, patch_y2 = self.crop_patches_info[idx]

        # patches
        height_icecap = self.height_map_icecap_tensor[:, patch_x1:patch_x2, patch_y1:patch_y2]
        bed_elevation = self.bedmachine[:, patch_x1:patch_x2, patch_y1:patch_y2]
        ice_velocity_x = self.ice_velocity_x_tensor[:, patch_x1:patch_x2, patch_y1:patch_y2]
        ice_velocity_y = self.ice_velocity_y_tensor[:, patch_x1:patch_x2, patch_y1:patch_y2]

        # Ensure all patches are of the same size
        assert height_icecap.shape == (1, self.patch_size, self.patch_size), f"Patch height icecap shape mismatch: {height_icecap.shape}"
        assert bed_elevation.shape == (1, self.patch_size, self.patch_size), f"Patch bed elevation shape mismatch: {bed_elevation.shape}"
        assert ice_velocity_x.shape == (1, self.patch_size, self.patch_size), f"Patch ice velocity x shape mismatch: {ice_velocity_x.shape}"
        assert ice_velocity_y.shape == (1, self.patch_size, self.patch_size), f"Patch ice velocity y shape mismatch: {ice_velocity_y.shape}"

        velocity = torch.cat((ice_velocity_x, ice_velocity_y), dim=0)

        height_icecap_lr = self.downsample_transform(height_icecap)
        bed_elevation_lr = self.downsample_transform(bed_elevation)
        velocity_lr = self.downsample_transform(velocity)

        hr_snow_accumulation = torch.rand((1, self.patch_size, self.patch_size))
        lr_snow_accumulation = self.downsample_transform(hr_snow_accumulation)


        return {
            'lr_height_icecap': height_icecap_lr,
            'hr_height_icecap': height_icecap,
            'lr_bed_elevation': bed_elevation_lr,
            'hr_bed_elevation': bed_elevation,
            'lr_velocity': velocity_lr,
            'hr_velocity': velocity,
            'hr_snow_accumulation': hr_snow_accumulation,
            'lr_snow_accumulation': lr_snow_accumulation
        }

    def plot_crop_overlay(self, idx, x, y, h, w, start_row, start_col):
        """Plots the original image with the crop and patch overlaid."""
        plt.figure(figsize=(10, 8))

        im = self.bedmachine.squeeze(0)

        new_height = im.shape[0] // 10
        new_width = im.shape[1] // 10

        im = F.interpolate(
            im.unsqueeze(0).unsqueeze(0),
            size=(new_height, new_width),
            mode='nearest'
        ).squeeze(0).squeeze(0)

        plt.imshow(im.squeeze(0).numpy(), cmap='terrain')

        crop_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', label='Crop')
        plt.gca().add_patch(crop_rect)

        patch_rect = patches.Rectangle((x + start_col, y + start_row), self.patch_size, self.patch_size, linewidth=2, edgecolor='b', facecolor='none', label='Patch')
        plt.gca().add_patch(patch_rect)

        plt.title(f"Crop {idx} Overlay")
        plt.legend()
        plt.savefig(f"figures/crop_{idx}_overlay.png")
        plt.close()

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    dataset = ArcticDataloader(
        bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
        arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
        snow_accumulation_path="data/Snow_acc/...",
        true_crops_folder="data/true_crops"
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for running it
    dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    print(f"DataLoader created with {len(dataloader)} batches")

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        for key, value in batch.items():
            print(batch.size)
            print(f"{key}: {value.shape}")