import os
import tarfile
import torch
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
        self.mass_balance = torch.tensor(self.mass_balance.values.astype(np.float32))

        self.hillshade_path = hillshade_path
        self.hillshade_tensor = self.read_hillshade_data()
        
        if region is None:
            self.true_crops = self.load_crops(path=true_crops)
            self.bedmachine_crops = self.load_crops(path=bedmachine_crops)
            self.arcticdem_crops = self.load_crops(path=arcticdem_crops)
        else:
            self.true_crops = self.load_crops(file_name="projected_crops.csv", use_coordinates=True)
            self.bedmachine_crops = self.load_crops(file_name="original_crops.csv", use_coordinates=True)
            self.arcticdem_crops = self.load_crops(file_name="arcticdem100_crops.csv", use_coordinates=True)

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
        aligned = data.rio.reproject_match(self.ice_velocity_data['land_ice_surface_easting_velocity'])
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

    dataset = ArcticDataset(region=True)

    batch_size = 128
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    def has_nan(tensor):
        return torch.isnan(tensor).any().item()

    for i, batch in enumerate(dataloader):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if has_nan(value):
                    print(f"NaN detected in batch {i} in key: {key}")
            elif isinstance(value, dict):
                continue  # crops dict, skip
            else:
                print(f"Unexpected data type for key '{key}' in batch {i}: {type(value)}")
        
        print(f"Dataloader created with {len(dataloader)} batches")

    bedmachine = dataset.bedmachine_projected.squeeze(0)

    plt.figure(figsize=(10, 8))
    plt.imshow(bedmachine, cmap="terrain", origin="upper")
    plt.colorbar(label="Bed Elevation (m)")

    for j, batch in enumerate(dataloader):
        for i in range(len(batch['hr_bed_elevation'])):
            y1, x1, y2, x2 = dataset.true_crops[i]
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c='r')

    plt.title("Visualization of Crop Locations over BedMachine Elevation")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig("figures/crop_locations.png", dpi=500)
    plt.close()

    if len(dataloader) == 1:
        batch = next(iter(dataloader))
        
        # Define layout for all plots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 4, figure=fig)
        
        # Dictionary mapping data types to their plot positions and titles
        plot_config = {
            'height_icecap': {'pos': gs[0, 0], 'title': 'ArcticDEM'},
            'lr_bed_elevation': {'pos': gs[0, 1], 'title': 'Low-res BedMachine'},
            'hr_bed_elevation': {'pos': gs[0, 2], 'title': 'High-res BedMachine'},
            'velocity_x': {'pos': gs[1, 0], 'title': 'Velocity (East-West)'},
            'velocity_y': {'pos': gs[1, 1], 'title': 'Velocity (North-South)'},
            'mass_balance': {'pos': gs[0, 3], 'title': 'Mass Balance'},
            'hillshade': {'pos': gs[1, 2], 'title': 'Flow-aware Hillshade'},
            'lr_arcticdem' : {'pos': gs[1, 3], 'title': 'Low-res ArcticDEM'},
        }

        # Create canvases and plot
        for img_type, config in plot_config.items():
            n_images = len(batch['velocity'])
            grid_size = int(np.ceil(np.sqrt(n_images)))

            if img_type in ['velocity_x', 'velocity_y']:

                ax = fig.add_subplot(config['pos'])
                
                # Create canvases for both components
                canvas_x = np.zeros((grid_size * batch['velocity'].shape[2], 
                                    grid_size * batch['velocity'].shape[3]))
                canvas_y = np.zeros_like(canvas_x)
                
                for idx in range(n_images):
                    i, j = idx // grid_size, idx % grid_size
                    y_start = i * batch['velocity'].shape[2]
                    y_end = (i + 1) * batch['velocity'].shape[2]
                    x_start = j * batch['velocity'].shape[3]
                    x_end = (j + 1) * batch['velocity'].shape[3]
                    
                    # Convert tensors to numpy arrays safely
                    x_data = batch['velocity'][idx][0].squeeze().cpu().detach().numpy()
                    y_data = batch['velocity'][idx][1].squeeze().cpu().detach().numpy()
                    canvas_x[y_start:y_end, x_start:x_end] = x_data.copy()
                    canvas_y[y_start:y_end, x_start:x_end] = y_data.copy()

                im = ax.imshow(canvas_x, cmap='viridis') if img_type == 'velocity_x' else ax.imshow(canvas_y, cmap='terrain')
                ax.set_title(config['title'])
                plt.colorbar(im, ax=ax)

            else:
                ax = fig.add_subplot(config['pos'])
                
                # Create and fill canvas
                canvas = np.zeros((grid_size * batch[img_type].shape[2], 
                                grid_size * batch[img_type].shape[3]))
                
                for idx in range(n_images):
                    i, j = idx // grid_size, idx % grid_size
                    y_start = i * batch[img_type].shape[2]
                    y_end = (i + 1) * batch[img_type].shape[2]
                    x_start = j * batch[img_type].shape[3]
                    x_end = (j + 1) * batch[img_type].shape[3]
                    
                    # Convert tensor to numpy array safely
                    img_data = batch[img_type][idx].squeeze().cpu().detach().numpy()
                    canvas[y_start:y_end, x_start:x_end] = img_data.copy()
                    
                im = ax.imshow(canvas, cmap='terrain')
                ax.set_title(config['title'])
                plt.colorbar(im, ax=ax)
            
            ax.axis('off')

        plt.suptitle('Overview of All Input Data Types', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('figures/batch_examples/glacier/all_inputs_overview.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        for i, batch in enumerate(dataloader):
            
            image_types = ['height_icecap', 'lr_bed_elevation', 'hr_bed_elevation', 'velocity', 'velocity', 'mass_balance', 'hillshade']
            titles = ["ArcticDem", "Low-res BedMachine",  "High-res BedMachine", "Velocity east/west", "Velocity north/south", "Mass balance", "Flow aware hillshade"]
            for j in range(batch_size):
                fig, axes = plt.subplots(1, len(image_types), figsize=(20, 5))
                
                for ax, img_type, title in zip(axes, image_types, titles):
                    if img_type == 'velocity':
                        img_x = batch[img_type][j][0].squeeze(0).numpy()
                        img_y = batch[img_type][j][1].squeeze(0).numpy()
                        ax.imshow(img_x, cmap="terrain")
                        ax.set_title(title + " X")
                        ax.axis("off")
                        ax.imshow(img_y)
                        ax.set_title(title + " Y")
                        ax.axis("off")
                    else:
                        img = batch[img_type][j].squeeze(0).numpy()
                        ax.imshow(img, cmap="terrain")
                        ax.set_title(title)
                        ax.axis("off")

                fig.savefig(f"figures/batch_examples/glacier/batch_of_crops_{j}.png", dpi=150)
                plt.close()
                if j == 30:
                    break
            break


