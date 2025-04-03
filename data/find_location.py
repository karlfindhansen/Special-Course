import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

class CroppedAreaGenerator:
    def __init__(self, bedmachine_path, ice_velocity_path, mass_balance_path, precise=True, crop_size=11, downscale=False):
        """Initializes the CroppedAreaGenerator with paths, processing settings, and data loading."""
        self.bedmachine_path = bedmachine_path
        self.ice_velocity_path = ice_velocity_path
        self.mass_balance_path = mass_balance_path
        self.precise = precise
        self.crop_size = crop_size
        self.downscale = downscale

        self.bedmachine_data = xr.open_dataset(self.bedmachine_path)
        self.ice_velocity_data = xr.open_dataset(self.ice_velocity_path)
        self.snow_accumulation_rate = xr.open_dataset(self.mass_balance_path)

        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)
        self.snow_accumulation_rate.rio.write_crs("EPSG:3413", inplace=True)

        self.bed_tensor, self.mask_tensor, self.ice_velocity_tensor, self.mass_balance_tensor, self.transform_info = self._load_and_preprocess_data()
                
        self.valid_indices = self._find_valid_crop_indices()

    def _load_and_preprocess_data(self):
        original_bed = self.bedmachine_data["bed"]
        original_errbed = self.bedmachine_data["errbed"]
        snow_acc_rate = self.snow_accumulation_rate["band_data"]

        bed_reprojected = original_bed.rio.reproject_match(self.ice_velocity_data)
        errbed_reprojected = original_errbed.rio.reproject_match(self.ice_velocity_data)
        snow_acc_rate_reprojected = snow_acc_rate.rio.reproject_match(self.ice_velocity_data)

        reproj_transform = bed_reprojected.rio.transform()

        bed_tensor = torch.tensor(bed_reprojected.values.astype(np.float32))
        errbed_tensor = torch.tensor(errbed_reprojected.values.astype(np.float32))
        mass_balance_tensor = torch.tensor(snow_acc_rate_reprojected.values.astype(np.float32)).squeeze(0)

        ice_velocity = self.ice_velocity_data["land_ice_surface_easting_velocity"]
        ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)

        mask_tensor = (errbed_tensor < 20).float() if self.precise else (errbed_tensor > 11).float()

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
                if occupied[i:i + self.crop_size, j:j + self.crop_size].any():
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

        plt.savefig(os.path.join(output_path, "crops_overlay.png"), dpi=300)
        plt.close()
        print(f"Overlay image saved to '{output_path}'")


if __name__ == '__main__':

    bedmachine_path = "data/inputs/Bedmachine/BedMachineGreenland-v5.nc"
    velocity_path = "data/inputs/Ice_velocity/Promice_AVG5year.nc"
    mass_balance_path = "data/inputs/Snow_acc/snow_acc_rate.tif"

    crop_generator = CroppedAreaGenerator(bedmachine_path, velocity_path, mass_balance_path, crop_size=121) 
    cropped_areas = crop_generator.generate_and_save_crops()

    if cropped_areas:
        print(f"Generated {len(cropped_areas)} cropped areas.")
        crop_generator.overlay_crops_on_mask()
    
