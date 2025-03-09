import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import sys

sys.path.append('src')
sys.path.append('src/Model')

from GeneratorModel import GeneratorModel

def read_icecap_height_data(arcticdem_path):
    """ Reads ArcticDEM data as a tensor. """
    arcticdem_data = rioxarray.open_rasterio(arcticdem_path)
    arcticdem_data.rio.write_crs("EPSG:3413", inplace=True)
    return arcticdem_data

def align_to_velocity(data):
    """ Reprojects and aligns data to match the velocity grid. """
    aligned = data.rio.reproject_match(ice_velocity_data['land_ice_surface_easting_velocity'])
    return torch.tensor(aligned.values.astype(np.float32))

def create_nxn_crops(x1, y1, x2, y2, crop_size=11):
    crops = []
    for i in range(y1, y2, crop_size):
        for j in range(x1, x2, crop_size):
            if i + crop_size <= y2 and j + crop_size <= x2:
                crops.append((i, j, i + crop_size, j + crop_size))
    return crops

model = os.path.join("res", "long_train", "best_generator.pth")
crop_size = 11
crop_path_org = os.path.join("data", "true_crops", "large_crops", "original_crops.csv")
crop_path_proj = os.path.join("data", "true_crops", "large_crops", "projected_crops.csv")

bedmachine_path = os.path.join("data", "Bedmachine", "BedMachineGreenland-v5.nc")
arcticdem_path= os.path.join("data", "arcticdem_extracted", "arcticdem_mosaic_500m_v4.1_dem.tif")
ice_velocity_path = os.path.join("data", "Ice_velocity", "Promice_AVG5year.nc")

generator_model = GeneratorModel()
generator_model.load_state_dict(torch.load(model, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')))

bedmachine_data = xr.open_dataset(bedmachine_path)
bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)

ice_velocity_data = xr.open_dataset(ice_velocity_path)
ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

height_map_icecap_data = read_icecap_height_data(arcticdem_path)
height_map_icecap_tensor = align_to_velocity(height_map_icecap_data)

ice_velocity_x_tensor = align_to_velocity(ice_velocity_data['land_ice_surface_easting_velocity'])
ice_velocity_y_tensor = align_to_velocity(ice_velocity_data['land_ice_surface_northing_velocity'])

bedmachine_projected = align_to_velocity(bedmachine_data['bed']).unsqueeze(0)

bed_elevation_hr = torch.tensor(bedmachine_data['bed'].values.astype(np.float32)).unsqueeze(0)

crop_path_org = pd.read_csv(crop_path_org)
crop_path_proj = pd.read_csv(crop_path_proj)

y_1_proj, x_1_proj, y_2_proj, x_2_proj = crop_path_proj.iloc[0]['y_1'], crop_path_proj.iloc[0]['x_1'], crop_path_proj.iloc[0]['y_2'], crop_path_proj.iloc[0]['x_2']
y_1_org, x_1_org, y_2_org, x_2_org = crop_path_org.iloc[0]['y_1'], crop_path_org.iloc[0]['x_1'], crop_path_org.iloc[0]['y_2'], crop_path_org.iloc[0]['x_2']

crops_proj = create_nxn_crops(x_1_proj, y_1_proj, x_2_proj, y_2_proj, crop_size=11)
crops_org = create_nxn_crops(x_1_org, y_1_org, x_2_org, y_2_org, crop_size=36)

grid_size = (11, 11)
stitched_image_generated = np.zeros((grid_size[0] * 36, grid_size[1] * 36))

generator_model.eval()
with torch.no_grad():
    for i, ((y1, x1, y2, x2),(y1_org, x1_org, y2_org, x2_org)) in enumerate(zip(crops_proj, crops_org)):
        bed_machine_lr = bedmachine_projected[:, y1:y2, x1:x2].unsqueeze(0)
        ice_velocity_x = ice_velocity_x_tensor[:, y1:y2, x1:x2]
        ice_velocity_y = ice_velocity_y_tensor[:, y1:y2, x1:x2]
        height_map_icecap = height_map_icecap_tensor[:, y1:y2, x1:x2].unsqueeze(0)
        bed_elevation_hr_1 = bed_elevation_hr[:, y1_org:y2_org, x1_org:x2_org].unsqueeze(0)

        velocity = torch.cat((ice_velocity_x, ice_velocity_y), dim=0).unsqueeze(0)
        snow_accumulation = torch.rand((1, crop_size, crop_size)).unsqueeze(0)

        
        output = generator_model(bed_machine_lr, height_map_icecap, velocity, snow_accumulation)
        output_image = output.squeeze(0).squeeze(0).cpu().numpy()

        # Compute row and column position
        row = i // grid_size[1]
        col = i % grid_size[1]

        # Place the image in the stitched canvas
        stitched_image_generated[row * 36:(row + 1) * 36, col * 36:(col + 1) * 36] = output_image


be_hr = bed_elevation_hr[:, y_1_org:y_2_org, x_1_org:x_2_org].squeeze(0).squeeze(0).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(20, 10)) 

axes[0].imshow(stitched_image_generated, cmap="terrain")
axes[0].axis("off")
axes[0].set_title("Stitched Output")

axes[1].imshow(be_hr, cmap="terrain")
axes[1].axis("off")
axes[1].set_title("High-Resolution Bed Elevation")

save_dir = os.path.join("comparison")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "stitched_and_bed_elevation.png"), bbox_inches="tight", dpi=300)
plt.show()