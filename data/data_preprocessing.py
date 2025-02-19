import os
import tarfile
import torch
import numpy as np
import xarray as xr
import rioxarray
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import plot_tensor

Image.MAX_IMAGE_PIXELS = None

class ArcticDataloader(Dataset):
    def __init__(self, bedmachine_path, arcticdem_path, ice_velocity_path):
        """ Initializes the dataset by loading and aligning data from NetCDF and GeoTIFF files. """

        self.bedmachine_data = xr.open_dataset(bedmachine_path)

        # Ensure BedMachine CRS is set (Polar Stereographic for Greenland)
        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)

        self.ice_velocity_data = xr.open_dataset(ice_velocity_path)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

        # Extract ArcticDEM from .tar file
        extracted_path = "data/arcticdem_extracted"
        os.makedirs(extracted_path, exist_ok=True)
        with tarfile.open(arcticdem_path) as tar:
            tar.extractall(path=extracted_path, filter="fully_trusted")

        tif_file = "arcticdem_mosaic_500m_v4.1_dem.tif"
        self.arcticdem_tif_path = os.path.join(extracted_path, tif_file)

        # Read and align ArcticDEM data
        self.height_map_icecap_tensor = self.read_icecap_height_data()

        # Align Ice Velocity data
        self.ice_velocity_x_tensor = self.align_to_bedmachine(
            self.ice_velocity_data['land_ice_surface_easting_velocity']
        )
        self.ice_velocity_y_tensor = self.align_to_bedmachine(
            self.ice_velocity_data['land_ice_surface_northing_velocity']
        )

        # Align ArcticDEM to BedMachine grid
        self.height_map_icecap_tensor = self.align_to_bedmachine(self.height_map_icecap_tensor)

        # Convert BedMachine 'errbed' to tensor
        self.bedmachine = torch.tensor(self.bedmachine_data['errbed'].values.astype(np.float32)).unsqueeze(0)

        # Process all tensors (flip and rotate for correct orientation)
        #self.height_map_icecap_tensor = self.process_tensor(self.height_map_icecap_tensor)
        self.bedmachine = self.process_tensor(self.bedmachine)
        #self.ice_velocity_x_tensor = self.process_tensor(self.ice_velocity_x_tensor)
        #self.ice_velocity_y_tensor = self.process_tensor(self.ice_velocity_y_tensor)

        #print(f"Shape of Surface elevation data: {self.height_map_icecap_tensor.size()}")
        #print(f"Shape of BedMachine data: {self.bedmachine.size()}")
        #print(f"Shape of Ice Velocity (East-West) data: {self.ice_velocity_x_tensor.size()}")
        #print(f"Shape of Ice Velocity (North-South) data: {self.ice_velocity_y_tensor.size()}")

    def read_icecap_height_data(self):
        """ Reads ArcticDEM data as a tensor. """
        arcticdem_data = rioxarray.open_rasterio(self.arcticdem_tif_path)
        arcticdem_data.rio.write_crs("EPSG:3413", inplace=True)
        return arcticdem_data

    def align_to_bedmachine(self, data):
        """ Reprojects and aligns data to match the BedMachine grid. """
        aligned = data.rio.reproject_match(self.bedmachine_data)
        return torch.tensor(aligned.values.astype(np.float32))

    def process_tensor(self, tensor):
        """ Rotates and flips the tensor to correct its orientation. """
        return torch.flip(torch.rot90(tensor, 2, dims=(0, 1)), dims=(1,))

    def __len__(self):
        """ Returns the number of available data points. """
        return 1

    def __getitem__(self, idx):
        """ Returns the aligned data as a dictionary. """
        return {
            'height_icecap': self.height_map_icecap_tensor,
            'bed_elevation': self.bedmachine,
            'ice_velocity_x': self.ice_velocity_x_tensor,
            'ice_velocity_y': self.ice_velocity_y_tensor,
        }

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    dataset = ArcticDataloader(
        bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
        arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for key, value in dataset[0].items():
        print(f"{key}: {value.size()}")

    for batch in dataloader:
        plot_tensor(batch['height_icecap'][0], "ArcticDEM Data", "height_icecap.png")
        plot_tensor(batch['bed_elevation'][0], "Bed Topography/Ice Thickness Error", "bed_elevation.png")
        plot_tensor(batch['ice_velocity_x'][0], "Ice Velocity (East-West)", "ice_velocity_x.png")
        plot_tensor(batch['ice_velocity_y'][0], "Ice Velocity (North-South)", "ice_velocity_y.png")
