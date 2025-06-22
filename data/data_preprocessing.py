import os
import torch
import numpy as np
import xarray as xr
import rioxarray
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

class ArcticDataset(Dataset):
    def __init__(self,
                 bedmachine_path   = os.path.join("data", "inputs", "Bedmachine", "BedMachineGreenland-v5.nc"),
                 arcticdem_path    = os.path.join("data", "inputs", "arcticdem", "arcticdem_mosaic_100m_v4.1_dem.tif"),
                 ice_velocity_path = os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc"),
                 mass_balance_path = os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif"),
                 hillshade_path    = os.path.join("data", "inputs", "hillshade", "macgregortest_flowalignedhillshade.tif"),
                 true_crops        = os.path.join("data","crops", "true_crops", "projected_crops.csv"),
                 bedmachine_crops  = os.path.join("data","crops", "true_crops","original_crops.csv"),
                 arcticdem_crops   = os.path.join("data", "crops", "true_crops","arcticdem100_crops.csv"),
                 region            = None
                 ):
        
        self.ice_velocity_data = xr.open_dataset(ice_velocity_path)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)
        self.ice_velocity_x_tensor = self.align_to_velocity(self.ice_velocity_data['land_ice_surface_easting_velocity'])
        self.ice_velocity_y_tensor = self.align_to_velocity(self.ice_velocity_data['land_ice_surface_northing_velocity'])

        self.bedmachine_data = xr.open_dataset(bedmachine_path)
        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)
        self.bedmachine_projected = self.align_to_velocity(self.bedmachine_data['bed']).unsqueeze(0)
        self.bed_elevation_hr = torch.tensor(self.bedmachine_data['bed'].values.astype(np.float32)).unsqueeze(0)

        self.arcticdem_path = arcticdem_path
        self.arcticdem_data = self.read_arcticdem()
        self.arcticdem_proj = self.align_to_velocity(self.arcticdem_data)
        self.arcicdem = torch.tensor(self.arcticdem_data.values.astype(np.float32))

        self.mass_balance = rioxarray.open_rasterio(mass_balance_path)
        self.mass_balance.rio.write_crs("EPSG:3413", inplace=True)
        self.mass_balance = self.align_to_velocity(self.mass_balance)

        self.hillshade_path = hillshade_path
        self.hillshade_tensor = self.read_hillshade_data()
        
        if region is None:
            glacier_name = None
            self.true_crops = self.load_crops(path=true_crops)
            self.bedmachine_crops = self.load_crops(path=bedmachine_crops)
            self.arcticdem_crops = self.load_crops(path=arcticdem_crops)
        else:
            glacier_name = "Kangerlussuaq" if isinstance(region, bool) else region
            self.true_crops = self.load_crops(file_name=f"projected_crops_{glacier_name}.csv", use_coordinates=True)
            self.bedmachine_crops = self.load_crops(file_name=f"original_crops_{glacier_name}.csv", use_coordinates=True)
            self.arcticdem_crops = self.load_crops(file_name=f"arcticdem100_crops_{glacier_name}.csv", use_coordinates=True)

        self.glacier_name = glacier_name
        self.crop_size = self.true_crops[0][2] - self.true_crops[0][0]
        self.bedmachine_crops_size = self.bedmachine_crops[0][2] - self.bedmachine_crops[0][0]

    
    def load_crops(self, path=None, file_name=None, use_coordinates=False):
        if use_coordinates:
            full_path = os.path.join("data", "crops", "coordinate_crops", file_name)
        else:
            full_path = path
        
        with open(full_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            return [list(map(int, row)) for row in reader]


    def read_arcticdem(self):
        arcticdem_data = rioxarray.open_rasterio(self.arcticdem_path)
        arcticdem_data.rio.write_crs("EPSG:3413", inplace=True)
        return arcticdem_data
    
    def read_hillshade_data(self):
        hillshade_data = rioxarray.open_rasterio(self.hillshade_path)
        hillshade_data.rio.write_crs("EPSG:3413", inplace=True)
        return torch.tensor(hillshade_data.values.astype(np.float32))

    def align_to_velocity(self, data):
        aligned = data.rio.reproject_match(self.ice_velocity_data)
        return torch.tensor(aligned.values.astype(np.float32))

    def __len__(self):
        return len(self.true_crops)

    def __getitem__(self, idx):
        y_1, x_1, y_2, x_2 = self.true_crops[idx]
        y_1_bed, x_1_bed, y_2_bed, x_2_bed = self.bedmachine_crops[idx]
        y_1_arcticdem, x_1_arcticdem, y_2_arcticdem, x_2_arcticdem = self.arcticdem_crops[idx]

        diff_y = y_2_bed - y_1_bed
        diff_x = x_2_bed - x_1_bed
        y_2_bed = y_2_bed - (diff_y - 72) if diff_y > 72 else y_2_bed
        x_2_bed = x_2_bed - (diff_x - 72) if diff_x > 72 else x_2_bed

        arcticdem_lr = self.arcticdem_proj[:, y_1:y_2, x_1:x_2]
        arcticdem = self.arcicdem[:, y_1_arcticdem:y_2_arcticdem, x_1_arcticdem:x_2_arcticdem]
        bed_elevation_lr = self.bedmachine_projected[:, y_1:y_2, x_1:x_2]
        bed_elevation_hr = self.bed_elevation_hr[:, y_1_bed:y_2_bed, x_1_bed:x_2_bed]
        
        ice_velocity_x = self.ice_velocity_x_tensor[:, y_1:y_2, x_1:x_2]
        ice_velocity_y = self.ice_velocity_y_tensor[:, y_1:y_2, x_1:x_2]
        mass_balance = self.mass_balance[:, y_1:y_2, x_1:x_2]
        hillshade = self.hillshade_tensor[:, y_1:y_2, x_1:x_2]

        velocity = torch.cat((ice_velocity_x, ice_velocity_y), dim=0)

        assert arcticdem.shape == (1, 110, 110), f"Patch height icecap shape mismatch: {arcticdem.shape}"
        assert arcticdem_lr.shape == (1, self.crop_size, self.crop_size), f"Arctic dem lr shape mismatch: {arcticdem_lr.shape}"
        assert mass_balance.shape == (1, self.crop_size, self.crop_size), f"Patch mass balance shape mismatch: {mass_balance.shape}"
        assert bed_elevation_lr.shape == (1, self.crop_size, self.crop_size), f"Patch bed elevation shape mismatch: {bed_elevation_lr.shape}"
        assert velocity.shape == (2, self.crop_size, self.crop_size), f"Patch ice velocity x shape mismatch: {ice_velocity_x.shape}"
        assert hillshade.shape == (1, self.crop_size, self.crop_size), f"Patch hillshade shape mismatch: {hillshade}"
        assert bed_elevation_hr.shape == (1, 72, 72), f"Patch bed elevation shape mismatch: {bed_elevation_hr.shape}"

        assert not torch.isnan(arcticdem).any(), "NaN values found in arcticdem tensor"
        assert not torch.isnan(arcticdem_lr).any(), "NaN values found in arcticdem_lr tensor"
        assert not torch.isnan(mass_balance).any(), "NaN values found in mass_balance tensor"
        assert not torch.isnan(bed_elevation_lr).any(), "NaN values found in bed_elevation_lr tensor"
        assert not torch.isnan(velocity).any(), "NaN values found in velocity tensor"
        assert not torch.isnan(hillshade).any(), "NaN values found in hillshade tensor"
        assert not torch.isnan(bed_elevation_hr).any(), "NaN values found in bed_elevation_hr tensor"

        crops = {'Projected'          : {'y_1':y_1 ,           'x_1':x_1,           'y_2':y_2,           'x_2':x_2},
                 'Original bed'       : {'y_1':y_1_bed ,       'x_1':x_1_bed,       'y_2':y_2_bed,       'x_2':x_2_bed},
                 'Original Arcticdem' : {'y_1':y_1_arcticdem , 'x_1':x_1_arcticdem, 'y_2':y_2_arcticdem, 'x_2':x_2_arcticdem}
                 }
        
        return {
            'height_icecap': arcticdem,
            'lr_bed_elevation': bed_elevation_lr,
            'hr_bed_elevation': bed_elevation_hr,
            'velocity': velocity,
            'mass_balance': mass_balance,
            'hillshade': hillshade,
            'lr_arcticdem' : arcticdem_lr,
            'crops' : crops
        }

if __name__ == "__main__":
    dataset = ArcticDataset()
    batch_size = 256
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
