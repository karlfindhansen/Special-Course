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
    def __init__(self, bedmachine_path, ice_velocity_path, precise=True, crop_size=11, downscale=False):
        """Initializes the CroppedAreaGenerator with paths, processing settings, and data loading."""
        self.bedmachine_path = bedmachine_path
        self.ice_velocity_path = ice_velocity_path
        self.precise = precise
        self.crop_size = crop_size
        self.downscale = downscale

        # Load datasets
        self.bedmachine_data = xr.open_dataset(self.bedmachine_path)
        self.ice_velocity_data = xr.open_dataset(self.ice_velocity_path)

        # Set CRS (Coordinate Reference System)
        self.bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)
        self.ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

        # Process the data and find valid crop locations
        self.bed_tensor, self.mask_tensor, self.ice_velocity_tensor, self.transform_info = self._load_and_preprocess_data()
        self.valid_indices = self._find_valid_crop_indices()

    def _load_and_preprocess_data(self):
        """Loads and reprojects datasets, and creates corresponding tensors."""
        original_bed = self.bedmachine_data["bed"]
        original_errbed = self.bedmachine_data["errbed"]

        orig_transform = original_bed.rio.transform()
        bed_reprojected = original_bed.rio.reproject_match(self.ice_velocity_data)
        errbed_reprojected = original_errbed.rio.reproject_match(self.ice_velocity_data)
        reproj_transform = bed_reprojected.rio.transform()

        # Convert to PyTorch tensors
        bed_tensor = torch.tensor(bed_reprojected.values.astype(np.float32))
        errbed_tensor = torch.tensor(errbed_reprojected.values.astype(np.float32))

        ice_velocity = self.ice_velocity_data["land_ice_surface_easting_velocity"]
        ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)

        # Create mask based on precision setting
        mask_tensor = (errbed_tensor < 11).float() if self.precise else (errbed_tensor > 11).float()

        # Store transformation information
        transform_info = {
            "original_shape": original_bed.shape,
            "reprojected_shape": bed_reprojected.shape,
            "original_transform": orig_transform,
            "reprojected_transform": reproj_transform,
            "original_dims": {"x": original_bed.x.values, "y": original_bed.y.values},
            "reprojected_dims": {"x": bed_reprojected.x.values, "y": bed_reprojected.y.values},
        }

        # Apply downscaling if enabled
        if self.downscale:
            factor = 5
            new_size = (mask_tensor.shape[0] // factor, mask_tensor.shape[1] // factor)

            def downscale(tensor):
                return F.interpolate(
                    tensor.unsqueeze(0).unsqueeze(0), size=new_size, mode="nearest"
                ).squeeze(0).squeeze(0)

            mask_tensor = downscale(mask_tensor)
            bed_tensor = downscale(bed_tensor)
            ice_velocity_tensor = downscale(ice_velocity_tensor)

            transform_info.update({"downscale_factor": factor, "downscaled_shape": new_size})

        return bed_tensor, mask_tensor, ice_velocity_tensor, transform_info

    def __len__(self):
        return len(self.valid_indices)

    def _projected_to_original_coords(self, y, x):
        """Converts reprojected coordinates to original BedMachine coordinates."""
        reproj_y, reproj_x = self.transform_info["reprojected_dims"]["y"][y], self.transform_info["reprojected_dims"]["x"][x]
        orig_y_idx = np.abs(self.transform_info["original_dims"]["y"] - reproj_y).argmin()
        orig_x_idx = np.abs(self.transform_info["original_dims"]["x"] - reproj_x).argmin()
        return orig_y_idx, orig_x_idx

    def _find_valid_crop_indices(self):
        """Finds valid crop areas based on mask constraints."""
        h, w = self.mask_tensor.shape
        occupied = np.zeros((h, w), dtype=bool)
        valid_crops_info = []
        edge_margin = 5
        crop_count = 0

        tqdm_bar = tqdm(range(edge_margin, h - self.crop_size - edge_margin + 1), desc="Finding valid crops")
        for i in tqdm_bar:
            for j in range(edge_margin, w - self.crop_size - edge_margin + 1):
                if occupied[i:i + self.crop_size, j:j + self.crop_size].any():
                    continue

                crop_mask = self.mask_tensor[i:i + self.crop_size, j:j + self.crop_size]
                crop_bed = self.bed_tensor[i:i + self.crop_size, j:j + self.crop_size]
                crop_velocity = self.ice_velocity_tensor[i:i + self.crop_size, j:j + self.crop_size]

                if torch.all(crop_mask == 1) and torch.all(crop_bed > 0) and torch.all(~torch.isnan(crop_velocity)):
                    end_row, end_col = i + self.crop_size, j + self.crop_size
                    orig_y1, orig_x1 = self._projected_to_original_coords(i, j)
                    orig_y2, orig_x2 = self._projected_to_original_coords(end_row, end_col)

                    valid_crops_info.append({"projected": [i, j, end_row, end_col], "original": [orig_y1, orig_x1, orig_y2, orig_x2]})
                    occupied[i:end_row, j:end_col] = True
                    crop_count += 1
                    tqdm_bar.set_postfix(total_crops=crop_count)

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
        """Generates and saves crop locations to CSV files."""
        if not self.valid_indices:
            return []

        output_dir = os.path.join(
            "data",
            "true_crops" if self.precise else "unprecise_crops",
            "large_crops" if self.crop_size == 121 else ""
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
        """Overlays the cropped areas on the mask image."""
        os.makedirs(output_path, exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(self.mask_tensor.numpy(), cmap="terrain")

        for crop in self.valid_indices:
            y1, x1, y2, x2 = crop["projected"]
            rect = patches.Rectangle((x1, y1), self.crop_size, self.crop_size, linewidth=1, edgecolor="r", facecolor="none")
            plt.gca().add_patch(rect)

        plt.title("Cropped Areas on Mask")
        plt.savefig(os.path.join(output_path, "crops_overlay.png"), dpi=300)
        plt.close()
        print(f"Overlay image saved to '{output_path}'")



if __name__ == '__main__':

    bedmachine_path = "data/Bedmachine/BedMachineGreenland-v5.nc"
    velocity_path = "data/Ice_velocity/Promice_AVG5year.nc"

    crop_generator = CroppedAreaGenerator(bedmachine_path, velocity_path, crop_size=121) 
    cropped_areas = crop_generator.generate_and_save_crops()

    if cropped_areas:
        print(f"Generated {len(cropped_areas)} cropped areas.")
        crop_generator.overlay_crops_on_mask()

