import os
import tarfile
import torch
import numpy as np
import xarray as xr
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None  

class ArcticDEMDataset(Dataset):
    def __init__(self, bedmachine_path, arcticdem_path, ice_velocity_path):
        """ Initializes the dataset by loading data from NetCDF and GeoTIFF files. """
        
        self.bedmachine_data = xr.open_dataset(bedmachine_path)
        self.ice_velocity_data = xr.open_dataset(ice_velocity_path)
        
        # Extract and read ArcticDEM data
        extracted_path = "data/arcticdem_extracted"
        os.makedirs(extracted_path, exist_ok=True)
        with tarfile.open(arcticdem_path) as tar:
            tar.extractall(path=extracted_path, filter="fully_trusted")
        
        tif_file = "arcticdem_mosaic_500m_v4.1_dem.tif"
        self.arcticdem_tif_path = os.path.join(extracted_path, tif_file)

        # Convert data to tensors
        self.height_map_icecap_tensor = self.read_arcticdem_data()
        self.errbed_tensor = torch.tensor(self.bedmachine_data['errbed'].values.astype(np.float32))
        self.ice_velocity_x_tensor = torch.tensor(self.ice_velocity_data['land_ice_surface_easting_velocity'].values.astype(np.float32)).squeeze(0)
        self.ice_velocity_y_tensor = torch.tensor(self.ice_velocity_data['land_ice_surface_northing_velocity'].values.astype(np.float32)).squeeze(0)

        # Process tensors to match orientation
        self.height_map_icecap_tensor = self.process_tensor(self.height_map_icecap_tensor)
        self.errbed_tensor = self.process_tensor(self.errbed_tensor)
        self.ice_velocity_x_tensor = self.process_tensor(self.ice_velocity_x_tensor)
        self.ice_velocity_y_tensor = self.process_tensor(self.ice_velocity_y_tensor)

    def read_arcticdem_data(self):
        """ Reads, crops, and converts ArcticDEM data to a PyTorch tensor. """
        with Image.open(self.arcticdem_tif_path) as img:
            width, height = img.size  
            bottom_fraction = 0.36
            side_fraction = 0.4
            width_size_east = 7000
            width_size_west = int(width * (1 - side_fraction))
            crop_box = (width_size_east, int(height * (1 - bottom_fraction)), width_size_west, height)

            cropped_img = img.crop(crop_box)
            arcticdem_data = np.array(cropped_img, dtype=np.float32)

        return torch.tensor(arcticdem_data[:-1, :-1])
    
    def process_tensor(self, tensor):
        """ Rotates and flips the tensor to correct its orientation. """
        return torch.flip(torch.rot90(tensor, 2, dims=(0, 1)), dims=(1,))
    
    def __len__(self):
        """ Returns the number of available data points. """
        return 1  # Since this dataset loads a single ArcticDEM instance

    def __getitem__(self, idx):
        """ Returns the data as a dictionary. """
        return {
            'arcticdem': self.height_map_icecap_tensor,
            'errbed': self.errbed_tensor,
            'ice_velocity_x': self.ice_velocity_x_tensor,
            'ice_velocity_y': self.ice_velocity_y_tensor,
        }


def plot_tensor(tensor, title, filename, cmap='gray'):
    """ Generic function to plot PyTorch tensors and save the figure. """
    plt.figure(figsize=(10, 8))
    plt.imshow(tensor.cpu().numpy(), cmap=cmap, origin='lower') 
    plt.colorbar(label=title)
    plt.title(title)
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    dataset = ArcticDEMDataset(
        bedmachine_path="data/BedMachineGreenland-v5.nc",
        arcticdem_path="data/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/dataverse_files/Promice_AVG5year.nc"
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for key, value in dataset[0].items():
        print(f"{key}: {value.size()}")
    
    # for batch in dataloader:
    #     plot_tensor(batch['arcticdem'][0], "ArcticDEM Data", "arcticdem_plot.png")
    #     plot_tensor(batch['errbed'][0], "Bed Topography/Ice Thickness Error", "errbed_plot.png")
    #     plot_tensor(batch['ice_velocity_x'][0], "Ice Velocity (East-West)", "ice_velocity_x.png")
    #     plot_tensor(batch['ice_velocity_y'][0], "Ice Velocity (North-South)", "ice_velocity_y.png")
