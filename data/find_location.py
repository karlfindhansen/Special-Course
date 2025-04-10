import os
import csv
import numpy as np
import torch
import rioxarray as rio
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import re
import pyproj
from affine import Affine

class CroppedAreaGenerator:
    def __init__(self, 
                 bedmachine_path = os.path.join("data", "inputs", "Bedmachine", "BedMachineGreenland-v5.nc"), 
                 ice_velocity_path= os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc"),
                 mass_balance_path= os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif"),
                 precise=True, crop_size=11, downscale=False, coordinates=None):
        """Initializes the CroppedAreaGenerator with paths, processing settings, and data loading."""
        self.bedmachine_path = bedmachine_path
        self.ice_velocity_path = ice_velocity_path
        self.mass_balance_path = mass_balance_path
        self.precise = precise
        self.crop_size = crop_size
        self.downscale = downscale

        self.bedmachine_data = xr.open_dataset(self.bedmachine_path)
        self.ice_velocity_data = xr.open_dataset(self.ice_velocity_path)
        self.mass_balance = rio.open_rasterio(self.mass_balance_path)

        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)
        self.mass_balance.rio.write_crs("EPSG:3413", inplace=True)

        self.bed_tensor, self.mask_tensor, self.ice_velocity_tensor, self.mass_balance_tensor, self.transform_info = self._load_and_preprocess_data()
        self.bed_tensor_transform = self.transform_info["reprojected_transform"]

        if coordinates is not None:
            lat, lon = coordinates
            source_crs = pyproj.CRS("EPSG:4326")
            target_crs = pyproj.CRS("EPSG:3413")
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            x_3413, y_3413 = transformer.transform(lon, lat)
            print(f"Original Coordinates (Lat, Lon): ({lat}, {lon})")
            print(f"Reprojected Point (X, Y in EPSG:3413): ({x_3413}, {y_3413})")
            print(f"Size of ice_velocity_tensor: {self.ice_velocity_tensor.shape}")

            fig, ax = plt.subplots(figsize=(10, 8))

            plot_data = self.ice_velocity_data['land_ice_surface_easting_velocity'].plot(
                ax=ax,
                cmap='viridis', 
                cbar_kwargs={'label': 'Ice Velocity (m/s)'},
            )

            ax.scatter(
                x_3413,      
                y_3413,       
                color='red',  
                s=100,       
                edgecolor='black', 
                marker='*',   
                label=f'Point ({lat:.2f}N, {lon:.2f}E)', # Label for legend
                zorder=10     # Ensure point is drawn on top
            )

            # --- 4. Customize Plot ---
            ax.set_title('Ice Velocity Data with Reprojected Point Location')
            # xarray.plot usually sets reasonable labels based on coordinate names and attrs
            # ax.set_xlabel("X Coordinate (EPSG:3413 meters)")
            # ax.set_ylabel("Y Coordinate (EPSG:3413 meters)")
            ax.legend() # Show the legend to identify the point
            ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio for map data
            plt.grid(True, linestyle='--', alpha=0.5) # Add a light grid
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            plt.show() # Display the plot
            plt.savefig("blablabla.png")
                
            exit()
            x, y, row, col = self.convert_latlon_to_pixel_ice_velocity(lat, lon)
            self.valid_indices = [self.find_largest_crop_around(col, row)]
        else:
            self.valid_indices = self._find_valid_crop_indices()


    
    def find_largest_crop_around(self, center_row, center_col, max_size=101):
        step = 0
        while True:
            top = center_row - step - 1
            bottom = center_row + step + 2
            left = center_col - step - 1
            right = center_col + step + 2

            if top < 0 or bottom > self.mask_tensor.shape[0] or left < 0 or right > self.mask_tensor.shape[1]:
                break

            crop_mask = self.mask_tensor[top:bottom, left:right]
            crop_bed = self.bed_tensor[top:bottom, left:right]
            crop_vel = self.ice_velocity_tensor[top:bottom, left:right]
            crop_mb = self.mass_balance_tensor[top:bottom, left:right]

            if torch.all(crop_mask == 1) and torch.all(crop_bed > 0) and torch.all(~torch.isnan(crop_vel)) and torch.all(~torch.isnan(crop_mb)):
                step += 1
                continue
            else:
                break

        step -= 1
        top = center_row - step
        left = center_col - step
        bottom = center_row + step + 1
        right = center_col + step + 1

        return {
            "projected": [top, left, bottom, right],
            "original": self._projected_to_original_coords(top, left) + self._projected_to_original_coords(bottom, right)
        }

    def _load_and_preprocess_data(self):
        original_bed = self.bedmachine_data["bed"]
        original_errbed = self.bedmachine_data["errbed"]
        mass_balance =  self.mass_balance

        bed_reprojected = original_bed.rio.reproject_match(self.ice_velocity_data)
        errbed_reprojected = original_errbed.rio.reproject_match(self.ice_velocity_data)
        snow_acc_rate_reprojected = mass_balance.rio.reproject_match(self.ice_velocity_data)

        reproj_transform = bed_reprojected.rio.transform()

        bed_tensor = torch.tensor(bed_reprojected.values.astype(np.float32))
        errbed_tensor = torch.tensor(errbed_reprojected.values.astype(np.float32))
        mass_balance_tensor = torch.tensor(snow_acc_rate_reprojected.values.astype(np.float32)).squeeze(0)

        ice_velocity = self.ice_velocity_data["land_ice_surface_easting_velocity"]
        ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)

        self.uncertaincy_criteria = 30
        mask_tensor = (errbed_tensor < self.uncertaincy_criteria).float() if self.precise else (errbed_tensor >= self.uncertaincy_criteria).float()

        transform_info = {
            "original_shape": original_bed.shape,
            "reprojected_shape": bed_reprojected.shape,
            "original_transform": original_bed.rio.transform(),
            "reprojected_transform": reproj_transform,
            "original_dims": {"x": original_bed.x.values, "y": original_bed.y.values},
            "reprojected_dims": {"x": bed_reprojected.x.values, "y": bed_reprojected.y.values},
        }

        return bed_tensor, mask_tensor, ice_velocity_tensor, mass_balance_tensor, transform_info

    def __len__(self):
        return len(self.valid_indices)

    def _projected_to_original_coords(self, y, x):
        reproj_y, reproj_x = self.transform_info["reprojected_dims"]["y"][y], self.transform_info["reprojected_dims"]["x"][x]
        orig_y_idx = np.abs(self.transform_info["original_dims"]["y"] - reproj_y).argmin()
        orig_x_idx = np.abs(self.transform_info["original_dims"]["x"] - reproj_x).argmin()
        return orig_y_idx, orig_x_idx

    def _find_valid_crop_indices(self):
        h, w = self.mask_tensor.shape
        occupied = np.zeros((h, w), dtype=bool)
        valid_crops_info = []
        edge_margin = 5
        base_size = 11
        crop_count = 0

        tqdm_bar = tqdm(range(edge_margin, h - self.crop_size - edge_margin + 1), desc="Finding valid crops")
        for i in tqdm_bar:
            for j in range(edge_margin, w - self.crop_size - edge_margin + 1):
                if not self.precise and occupied[i:i + self.crop_size, j:j + self.crop_size].any():
                    continue

                crop_mask = self.mask_tensor[i:i + self.crop_size, j:j + self.crop_size]
                crop_bed = self.bed_tensor[i:i + self.crop_size, j:j + self.crop_size]
                crop_velocity = self.ice_velocity_tensor[i:i + self.crop_size, j:j + self.crop_size]
                crop_mass_balance = self.mass_balance_tensor[i:i+self.crop_size, j:j+self.crop_size]

                if torch.all(crop_mask == 1) and torch.all(crop_bed > 0) and torch.all(~torch.isnan(crop_velocity)) and torch.all(~torch.isnan(crop_mass_balance)):
                    end_row, end_col = i + self.crop_size, j + self.crop_size

                    if self.crop_size > base_size:
                        for sub_i in range(i, end_row, base_size):
                            for sub_j in range(j, end_col, base_size):
                                sub_end_row = min(sub_i + base_size, end_row)
                                sub_end_col = min(sub_j + base_size, end_col)
                                orig_y1, orig_x1 = self._projected_to_original_coords(sub_i, sub_j)
                                orig_y2, orig_x2 = self._projected_to_original_coords(sub_end_row, sub_end_col)
                                
                                valid_crops_info.append({
                                    "projected": [sub_i, sub_j, sub_end_row, sub_end_col],
                                    "original": [orig_y1, orig_x1, orig_y2, orig_x2]
                                })
                                occupied[sub_i:sub_end_row, sub_j:sub_end_col] = True
                                crop_count += 1

                    else:
                        orig_y1, orig_x1 = self._projected_to_original_coords(i, j)
                        orig_y2, orig_x2 = self._projected_to_original_coords(end_row, end_col)

                        valid_crops_info.append({"projected": [i, j, end_row, end_col], 
                                                 "original": [orig_y1, orig_x1, orig_y2, orig_x2]})
                        occupied[i:end_row, j:end_col] = True
                        crop_count += 1

                    tqdm_bar.set_postfix(total_crops=crop_count if self.crop_size == 11 else crop_count/self.crop_size)

        return valid_crops_info

    def _save_crops_to_csv(self, data, filename):
        """Saves crop coordinates to a CSV file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["y_1", "x_1", "y_2", "x_2"])
            for crop in data:
                writer.writerow(crop)

    def generate_and_save_crops(self):
        if not self.valid_indices:
            return []

        output_dir = os.path.join(
            "data",
            "crops",
            "true_crops" if self.precise else "unprecise_crops",
            "large_crops" if self.crop_size > 11 else ""
        ).rstrip(os.sep)
        if self.downscale:
            output_dir = os.path.join(output_dir, "downscaled")

        projected_csv = os.path.join(output_dir, "projected_crops.csv")
        original_csv = os.path.join(output_dir, "original_crops.csv")

        self._save_crops_to_csv([crop["projected"] for crop in self.valid_indices], projected_csv)
        self._save_crops_to_csv([crop["original"] for crop in self.valid_indices], original_csv)

        print(f"Saved projected crop coordinates: {projected_csv}")
        print(f"Saved original crop coordinates: {original_csv}")

        return self.valid_indices

    def overlay_crops_on_mask(self, output_path="figures/with_crops"):
        os.makedirs(output_path, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f"Cropped Areas Overlay with {self.uncertaincy_criteria} meters uncertaincy", fontsize=16)
        axes[0].imshow(self.mask_tensor.numpy(), cmap="terrain")
        for crop in self.valid_indices:
            y1, x1, y2, x2 = crop["projected"]
            rect = patches.Rectangle((x1, y1), self.crop_size, self.crop_size, linewidth=1, edgecolor="r")
            axes[0].add_patch(rect)
        axes[0].set_title("Cropped Areas on Mask")

        axes[1].imshow(self.mass_balance_tensor.numpy(), cmap="terrain")
        for crop in self.valid_indices:
            y1, x1, y2, x2 = crop["projected"]
            rect = patches.Rectangle((x1, y1), self.crop_size, self.crop_size, linewidth=1, edgecolor="r")
            axes[1].add_patch(rect)
        axes[1].set_title("Cropped Areas on Mass Balance")
        output_name = "crops_overlay.png" if self.precise else "unprecise_crops_overlay.png"
        plt.savefig(os.path.join(output_path, output_name), dpi=300)
        plt.close()
        print(f"Overlay image saved to '{output_path}'")


if __name__ == '__main__':

    coordinates = (68.38, 33.00)
    coordinates = (round(int(coordinates[0]) + (coordinates[0] % 1*100 / 60),4), round(int(coordinates[1]) + (coordinates[1] % 1*100 / 60),4))
    crop_generator = CroppedAreaGenerator(crop_size=11, precise=True, coordinates=coordinates) 
    cropped_areas = crop_generator.generate_and_save_crops()

    if cropped_areas:
        print(f"Generated {len(cropped_areas)} cropped areas.")
        crop_generator.overlay_crops_on_mask()
    
