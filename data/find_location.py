import xarray as xr
import torch
import os
import numpy as np
import rioxarray
import random
import csv
import shutil
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CroppedAreaGenerator:
    def __init__(self, bedmachine_path, ice_velocity_path, crop_size=11, num_crops=None, downscale=False):
        self.bedmachine_path = bedmachine_path
        self.ice_velocity_path = ice_velocity_path

        self.crop_size = crop_size
        self.num_crops = num_crops
        self.downscale = downscale  
        self.bed_tensor, self.mask_tensor, self.ice_velocity_tensor = self._load_and_preprocess_data()
        self.valid_indices = self._find_valid_crop_indices()

    def _load_and_preprocess_data(self):
        """ Load BedMachine and ice velocity data, projecting BedMachine to match ice velocity. """
        # Load ice velocity data first
        ice_velocity_data = xr.open_dataset(self.ice_velocity_path)
        ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

        # Load BedMachine data and reproject to match ice velocity
        bedmachine_data = xr.open_dataset(self.bedmachine_path)
        bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)

        bed_reprojected = bedmachine_data['bed'].rio.reproject_match(ice_velocity_data)
        errbed_reprojected = bedmachine_data['errbed'].rio.reproject_match(ice_velocity_data)

        bed_tensor = torch.tensor(bed_reprojected.values.astype(np.float32))
        errbed_tensor = torch.tensor(errbed_reprojected.values.astype(np.float32))

        # Keep ice velocity unchanged
        ice_velocity = ice_velocity_data['land_ice_surface_easting_velocity']
        ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)

        mask_tensor = (errbed_tensor < 11).float()
        
        if self.downscale:
            new_height = mask_tensor.shape[0] // 5
            new_width = mask_tensor.shape[1] // 5

            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0).squeeze(0)
            bed_tensor = F.interpolate(bed_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0).squeeze(0)
            ice_velocity_tensor = F.interpolate(ice_velocity_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0).squeeze(0)

        return bed_tensor, mask_tensor, ice_velocity_tensor
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self):
        return self.valid_indices

    def _find_valid_crop_indices(self):
        crops_counter = 0
        valid_crops_info = []
        h, w = self.mask_tensor.shape
        occupied = np.zeros((h, w), dtype=bool)
        edge_distance = 5

        tqdm_bar = tqdm(range(edge_distance, h - self.crop_size - edge_distance + 1), desc="Finding valid crops")

        for i in tqdm_bar:
            for j in range(edge_distance, w - self.crop_size - edge_distance + 1):
                if not np.any(occupied[i:i + self.crop_size, j:j + self.crop_size]):
                    crop_mask = self.mask_tensor[i:i + self.crop_size, j:j + self.crop_size]
                    crop_bed = self.bed_tensor[i:i + self.crop_size, j:j + self.crop_size]
                    crop_velocity = self.ice_velocity_tensor[i:i + self.crop_size, j:j + self.crop_size]
                    
                    if torch.all(crop_mask == 1) and torch.all(crop_bed > 0) and torch.all(~torch.isnan(crop_velocity)):
                        end_row = i + self.crop_size
                        end_col = j + self.crop_size
                        valid_crops_info.append([i, j, end_row, end_col])
                        occupied[i:end_row, j:end_col] = True
                        crops_counter += 1
                        tqdm_bar.set_postfix(total_crops=crops_counter)
        if len(valid_crops_info) > 10000:
            valid_crops_info = random.sample(population=valid_crops_info, k=10000)        
        return valid_crops_info

    def generate_and_save_crops(self):
        crops_indices = self.valid_indices
        if self.num_crops is None:
            crops_to_generate = len(crops_indices)
        else:
            crops_to_generate = min(self.num_crops, len(crops_indices))

        if crops_to_generate == 0:
            return []

        output_dir = os.path.join("data", "true_crops" if not self.downscale else "downscaled_true_crops")
        os.makedirs(output_dir, exist_ok=True)

        csv_file_path = os.path.join(output_dir, "selected_crops.csv")
        with open(csv_file_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["y_1", "x_1", "y_2", "x_2"]) 
            writer.writerows(crops_indices)

        print(f"Selected crops saved to CSV: {csv_file_path}")

        return crops_indices
        
    def overlay_crops_on_mask(self, output_path="figures"):
        """Overlays the generated crops on the mask image and saves it."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.mask_tensor.numpy(), cmap='terrain')

        for crop in self.valid_indices:
            y_1, x_1, y_2, x_2 = crop
            rect = patches.Rectangle((x_1, y_1), self.crop_size, self.crop_size, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.title("Mask with Cropped Areas")
        if self.downscale:
            plt.savefig(os.path.join(output_path, "downscaled_crops_overlay.png"), dpi=300)
        else:
            plt.savefig(os.path.join(output_path, "true_size_crops_overlay.png"), dpi=300)
        plt.close()
        print(f"Overlay image saved to '{output_path}'")


if __name__ == '__main__':

    # def align_to_bedmachine(bedmachine_data, data):
    #     """ Reprojects and aligns data to match the BedMachine grid. """
    #     aligned = data.rio.reproject_match(bedmachine_data)
    #     return aligned

    bedmachine_path = "data/Bedmachine/BedMachineGreenland-v5.nc"
    velocity_path = "data/Ice_velocity/Promice_AVG5year.nc"

    # bedmachine_data = xr.open_dataset(bedmachine_path)
    # bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)

    # ice_velocity_data = xr.open_dataset(velocity_path)
    # ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

    # ice_velocity_data = align_to_bedmachine(
    #         bedmachine_data['bed'],
    #         ice_velocity_data['land_ice_surface_easting_velocity']
    #     )


    crop_generator = CroppedAreaGenerator(bedmachine_path, velocity_path, downscale=False) 
    cropped_areas = crop_generator.generate_and_save_crops()
    #print(cropped_areas)

    if cropped_areas:
        print(f"Generated {len(cropped_areas)} cropped areas.")
        crop_generator.overlay_crops_on_mask()

