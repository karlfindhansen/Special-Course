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
from affine import Affine

class CroppedAreaGenerator:
    def __init__(self, bedmachine_path, ice_velocity_path, crop_size=11, num_crops=None, downscale=False):
        self.bedmachine_path = bedmachine_path
        self.ice_velocity_path = ice_velocity_path

        self.crop_size = crop_size
        self.num_crops = num_crops
        self.downscale = downscale  
        
        # Load the datasets
        self.bedmachine_data = xr.open_dataset(self.bedmachine_path)
        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)
        
        self.ice_velocity_data = xr.open_dataset(self.ice_velocity_path)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)
        
        # Process the data
        self.bed_tensor, self.mask_tensor, self.ice_velocity_tensor, self.transform_info = self._load_and_preprocess_data()
        self.valid_indices = self._find_valid_crop_indices()

    def _load_and_preprocess_data(self):
        """ Load BedMachine and ice velocity data, projecting BedMachine to match ice velocity. """
        # Keep original BedMachine grid information
        original_bed = self.bedmachine_data['bed']
        original_errbed = self.bedmachine_data['errbed']
        
        # Store the transform information
        orig_transform = original_bed.rio.transform()
        
        # Reproject BedMachine to match ice velocity
        bed_reprojected = original_bed.rio.reproject_match(self.ice_velocity_data)
        errbed_reprojected = original_errbed.rio.reproject_match(self.ice_velocity_data)
        
        # Store the reprojected transform
        reproj_transform = bed_reprojected.rio.transform()

        # Create tensors
        bed_tensor = torch.tensor(bed_reprojected.values.astype(np.float32))
        errbed_tensor = torch.tensor(errbed_reprojected.values.astype(np.float32))

        # Keep ice velocity unchanged
        ice_velocity = self.ice_velocity_data['land_ice_surface_easting_velocity']
        ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)

        mask_tensor = (errbed_tensor < 11).float()
        
        # Store original and reprojected shapes and transforms for inverse transformation
        transform_info = {
            'original_shape': original_bed.shape,
            'reprojected_shape': bed_reprojected.shape,
            'original_transform': orig_transform,
            'reprojected_transform': reproj_transform,
            'original_dims': {
                'x': original_bed.x.values,
                'y': original_bed.y.values
            },
            'reprojected_dims': {
                'x': bed_reprojected.x.values,
                'y': bed_reprojected.y.values
            }
        }
        
        if self.downscale:
            new_height = mask_tensor.shape[0] // 5
            new_width = mask_tensor.shape[1] // 5

            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0).squeeze(0)
            bed_tensor = F.interpolate(bed_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0).squeeze(0)
            ice_velocity_tensor = F.interpolate(ice_velocity_tensor.unsqueeze(0).unsqueeze(0), size=(new_height, new_width), mode='nearest').squeeze(0).squeeze(0)
            
            # Adjust transformation info for downscaling
            transform_info['downscale_factor'] = 5
            transform_info['downscaled_shape'] = (new_height, new_width)

        return bed_tensor, mask_tensor, ice_velocity_tensor, transform_info
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self):
        return self.valid_indices
    
    def _projected_to_original_coords(self, y, x):
        """Convert coordinates from the reprojected grid to the original BedMachine grid"""
        # Get the reprojected spatial coordinates
        reproj_y = self.transform_info['reprojected_dims']['y'][y]
        reproj_x = self.transform_info['reprojected_dims']['x'][x]
        
        # Find the closest indices in the original grid
        original_y_idx = np.abs(self.transform_info['original_dims']['y'] - reproj_y).argmin()
        original_x_idx = np.abs(self.transform_info['original_dims']['x'] - reproj_x).argmin()
        
        return original_y_idx, original_x_idx

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
                        
                        # Get original coordinates for top-left and bottom-right corners
                        orig_y1, orig_x1 = self._projected_to_original_coords(i, j)
                        orig_y2, orig_x2 = self._projected_to_original_coords(end_row, end_col)
                        
                        # Store both projected and original coordinates
                        valid_crops_info.append({
                            'projected': [i, j, end_row, end_col],
                            'original': [orig_y1, orig_x1, orig_y2, orig_x2]
                        })
                        
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

        # Save projected coordinates
        proj_csv_path = os.path.join(output_dir, "projected_crops.csv")
        with open(proj_csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["y_1", "x_1", "y_2", "x_2"]) 
            for crop in crops_indices:
                writer.writerow(crop['projected'])
        
        print(f"Projected coordinates saved to CSV: {proj_csv_path}")
        
        # Save original coordinates
        orig_csv_path = os.path.join(output_dir, "original_crops.csv")
        with open(orig_csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["y_1", "x_1", "y_2", "x_2"]) 
            for crop in crops_indices:
                writer.writerow(crop['original'])
        
        print(f"Original coordinates saved to CSV: {orig_csv_path}")

        return crops_indices
        
    def overlay_crops_on_mask(self, output_path="figures"):
        """Overlays the generated crops on the mask image and saves it."""
        os.makedirs(output_path, exist_ok=True)
        
        # Overlay on reprojected image
        plt.figure(figsize=(10, 8))
        plt.imshow(self.mask_tensor.numpy(), cmap='terrain')

        for crop in self.valid_indices:
            y_1, x_1, y_2, x_2 = crop['projected']
            rect = patches.Rectangle((x_1, y_1), self.crop_size, self.crop_size, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.title("Reprojected Mask with Cropped Areas")
        proj_filename = "downscaled_crops_overlay.png" if self.downscale else "true_size_crops_overlay.png"
        plt.savefig(os.path.join(output_path, proj_filename), dpi=300)
        plt.close()
        
        # Overlay on original BedMachine image
        original_bed = self.bedmachine_data['bed'].values
        plt.figure(figsize=(10, 8))
        plt.imshow(original_bed, cmap='terrain')
        
        for crop in self.valid_indices:
            orig_y1, orig_x1, orig_y2, orig_x2 = crop['original']
            crop_width = orig_x2 - orig_x1
            crop_height = orig_y2 - orig_y1
            rect = patches.Rectangle((orig_x1, orig_y1), crop_width, crop_height, 
                                    linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
        plt.title("Original BedMachine with Cropped Areas")
        orig_filename = "downscaled_crops_original_overlay.png" if self.downscale else "true_size_crops_original_overlay.png"
        plt.savefig(os.path.join(output_path, orig_filename), dpi=300)
        plt.close()
        
        print(f"Overlay images saved to '{output_path}'")


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

