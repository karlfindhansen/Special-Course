import os
import tarfile
import torch
import numpy as np
import xarray as xr
import rioxarray
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

class ArcticDataloader(Dataset):
    def __init__(self,
                 bedmachine_path,
                 arcticdem_path,
                 ice_velocity_path,
                 snow_accumulation_path,
                 true_crops = os.path.join("data", "true_crops", "projected_crops.csv"),
                 bedmachine_crops = os.path.join("data", "true_crops", "original_crops.csv")
                 ):
        
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
        self.height_map_icecap_tensor = self.align_to_velocity(self.height_map_icecap_data)

        self.ice_velocity_x_tensor = self.align_to_velocity(
            self.ice_velocity_data['land_ice_surface_easting_velocity']
        )
        
        self.ice_velocity_y_tensor = self.align_to_velocity(
            self.ice_velocity_data['land_ice_surface_northing_velocity']
        )

        self.bedmachine_projected = self.align_to_velocity(
            self.bedmachine_data['bed']
        ).unsqueeze(0)

        with open(true_crops, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            self.true_crops = [list(map(int, row)) for row in reader]

        with open(bedmachine_crops, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            self.bedmachine_crops = [list(map(int, row)) for row in reader]
        
        self.crop_size = self.true_crops[0][2] - self.true_crops[0][0]
        self.bedmachine_crops_size = self.bedmachine_crops[0][2] - self.bedmachine_crops[0][0]

        self.bed_elevation_hr = torch.tensor(self.bedmachine_data['bed'].values.astype(np.float32)).unsqueeze(0)

        
    def read_icecap_height_data(self):
        """ Reads ArcticDEM data as a tensor. """
        arcticdem_data = rioxarray.open_rasterio(self.arcticdem_tif_path)
        arcticdem_data.rio.write_crs("EPSG:3413", inplace=True)
        return arcticdem_data

    def align_to_velocity(self, data):
        """ Reprojects and aligns data to match the velocity grid. """
        aligned = data.rio.reproject_match(self.ice_velocity_data['land_ice_surface_easting_velocity'])
        return torch.tensor(aligned.values.astype(np.float32))
    
    def align_to_crop_to_bedmachine(self, data):
        """Reprojects and aligns data to match the original bedmachine grid."""
        aligned = data.rio.reproject_match(self.bedmachine_data['bed'])
        return torch.tensor(aligned.values.astype(np.float32))

    def __len__(self):
        """ Returns the number of available data points. """
        return len(self.true_crops)

    def __getitem__(self, idx):
        """ Returns a fixed-size patch from the crop. """
        y_1, x_1, y_2, x_2 = self.true_crops[idx]
        y_1_b, x_1_b, y_2_b, x_2_b = self.bedmachine_crops[idx]

        y_2_b -= 1 if (y_2_b - y_1_b) == 37 else y_2_b
        x_2_b -= 1 if (x_2_b - x_1_b) == 37 else x_2_b
        
        height_icecap = self.height_map_icecap_tensor[:, y_1:y_2, x_1:x_2]
        bed_elevation_lr = self.bedmachine_projected[:, y_1:y_2, x_1:x_2]
        bed_elevation_hr = self.bed_elevation_hr[:, y_1_b:y_2_b, x_1_b:x_2_b]
        ice_velocity_x = self.ice_velocity_x_tensor[:, y_1:y_2, x_1:x_2]
        ice_velocity_y = self.ice_velocity_y_tensor[:, y_1:y_2, x_1:x_2]

        snow_accumulation = torch.rand((1, self.crop_size, self.crop_size))

        velocity = torch.cat((ice_velocity_x, ice_velocity_y), dim=0)

        assert height_icecap.shape == (1, self.crop_size, self.crop_size), f"Patch height icecap shape mismatch: {height_icecap.shape}"
        assert snow_accumulation.shape == (1, self.crop_size, self.crop_size), f"Patch snow accumulations shape mismatch: {height_icecap.shape}"
        assert bed_elevation_lr.shape == (1, self.crop_size, self.crop_size), f"Patch bed elevation shape mismatch: {bed_elevation_lr.shape}"
        assert velocity.shape == (2, self.crop_size, self.crop_size), f"Patch ice velocity x shape mismatch: {ice_velocity_x.shape}"

        snow_accumulation = torch.rand((1, self.crop_size, self.crop_size))

        return {
            'height_icecap': height_icecap,
            'lr_bed_elevation': bed_elevation_lr,
            'hr_bed_elevation': bed_elevation_hr,
            'velocity': velocity,
            'snow_accumulation': snow_accumulation,
        }


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    dataset = ArcticDataloader(
        bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
        arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
        snow_accumulation_path="data/Snow_acc/...",
        true_crops="data/true_crops/selected_crops.csv"
    )

    # Split dataset into training and validation sets
    train_size = int(0.95 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)

    print(f"DataLoader created with {len(dataloader)} batches")

    bedmachine = dataset.bedmachine_projected.squeeze(0)

    plt.figure(figsize=(10, 8))
    plt.imshow(bedmachine, cmap="terrain", origin="upper")
    plt.colorbar(label="Bed Elevation (m)")

    index = 0
    for j, batch in enumerate(dataloader):
        for i in range(len(batch['hr_bed_elevation'])):
            index += 1
            y1, x1, y2, x2 = dataset.true_crops[index]
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c='r')

    plt.title("Visualization of Crop Locations over BedMachine Elevation")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig("figures/crop_locations.png", dpi=500)

    batch = next(iter(dataloader))
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))

    image_types = ['height_icecap', 'lr_bed_elevation', 'hr_bed_elevation', 'velocity', 'velocity', 'snow_accumulation']
    titles = ["Height Icecap", "Low-res Bed Elevation",  "High res Bed Elevation", "Velocity X", "Velocity Y", "Snow Accumulation"]

    for ax, img_type, title in zip(axes, image_types, titles):
        if img_type == 'velocity':
            img_x = batch[img_type][0][0].squeeze(0).numpy()
            img_y = batch[img_type][0][1].squeeze(0).numpy()
            ax.imshow(img_x, cmap="terrain")
            ax.set_title(title + " X")
            ax.axis("off")
            ax.imshow(img_y)
            ax.set_title(title + " Y")
            ax.axis("off")
        else:
            img = batch[img_type][0].squeeze(0).numpy()
            ax.imshow(img, cmap="terrain")
            ax.set_title(title)
            ax.axis("off")

    fig.savefig("figures/batch_of_crops.png", dpi=150)
