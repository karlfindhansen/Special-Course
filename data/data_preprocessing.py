import os
import tarfile
import torch
import numpy as np
import xarray as xr
import rioxarray
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split

Image.MAX_IMAGE_PIXELS = None
class ArcticDataloader(Dataset):
    def __init__(self, 
                 bedmachine_path, 
                 arcticdem_path, 
                 ice_velocity_path, 
                 snow_accumulation_path,
                 true_crops_folder):
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

        self.height_map_icecap_tensor = self.read_icecap_height_data()

        self.ice_velocity_x_tensor = self.align_to_bedmachine(
            self.ice_velocity_data['land_ice_surface_easting_velocity']
        )
        self.ice_velocity_y_tensor = self.align_to_bedmachine(
            self.ice_velocity_data['land_ice_surface_northing_velocity']
        )

        self.height_map_icecap_tensor = self.align_to_bedmachine(self.height_map_icecap_tensor)

        self.bedmachine = torch.tensor(self.bedmachine_data['bed'].values.astype(np.float32)).unsqueeze(0)

        # Load true crop coordinates
        self.true_crops = [np.load(os.path.join(true_crops_folder, file)) for file in os.listdir(true_crops_folder)]

    def read_icecap_height_data(self):
        """ Reads ArcticDEM data as a tensor. """
        arcticdem_data = rioxarray.open_rasterio(self.arcticdem_tif_path)
        arcticdem_data.rio.write_crs("EPSG:3413", inplace=True)
        return arcticdem_data

    def align_to_bedmachine(self, data):
        """ Reprojects and aligns data to match the BedMachine grid. """
        aligned = data.rio.reproject_match(self.bedmachine_data)
        return torch.tensor(aligned.values.astype(np.float32))

    def __len__(self):
        """ Returns the number of available data points. """
        return len(self.true_crops)

    def __getitem__(self, idx):
        """ Returns the aligned data as a dictionary. """
        crop = self.true_crops[idx]
        x, y, h, w = crop

        return {
            'height_icecap': self.height_map_icecap_tensor[y:y + h, x:x + w],
            'bed_elevation': self.bedmachine[y:y + h, x:x + w],
            'ice_velocity_x': self.ice_velocity_x_tensor[y:y + h, x:x + w],
            'ice_velocity_y': self.ice_velocity_y_tensor[y:y + h, x:x + w],
        }


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    true_crops_folder = "data/true_crops"
    dataset = ArcticDataloader(
        bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
        arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
        snow_accumulation_path="data/Snow_acc/...",
        true_crops_folder=true_crops_folder
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Example usage
    for batch in train_dataloader:
        print(batch['height_icecap'].shape)  # Should print torch.Size([128, h, w])
        break