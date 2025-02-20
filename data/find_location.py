import xarray as xr
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CroppedAreaGenerator:
    def __init__(self, bedmachine_path, crop_size=100, num_crops=None, downscale=False):
        self.bedmachine_path = bedmachine_path
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.downscale = downscale  
        self.bed_tensor, self.mask_tensor = self._load_and_preprocess_data()
        self.valid_indices = self._find_valid_crop_indices()

    def _load_and_preprocess_data(self):
        bedmachine_data = xr.open_dataset(self.bedmachine_path)
        errbed_tensor = torch.tensor(bedmachine_data['errbed'].values.astype(np.float32))
        bed_tensor = torch.tensor(bedmachine_data['bed'].values.astype(np.float32))

        mask_tensor = (errbed_tensor < 10).float()
        
        if self.downscale:
            new_height = mask_tensor.shape[0] // 10
            new_width = mask_tensor.shape[1] // 10

            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),  
                size=(new_height, new_width),
                mode='nearest'  
            ).squeeze(0).squeeze(0) # keep this if I want to test with smaller images to reduce time to generate crops.

        return bed_tensor, mask_tensor

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

                    if torch.all(crop_mask == 1) and torch.all(crop_bed > 0):
                        end_row = i + self.crop_size
                        end_col = j + self.crop_size
                        valid_crops_info.append([i, j, end_row, end_col])
                        occupied[i:end_row, j:end_col] = True
                        crops_counter += 1
                        tqdm_bar.set_postfix(total_crops=crops_counter)

        return valid_crops_info

    def generate_and_save_crops(self):
        if self.num_crops is None:
            crops_to_generate = len(self.valid_indices)
        else:
            crops_to_generate = min(self.num_crops, len(self.valid_indices))

        if crops_to_generate == 0:
            return []

        def crop_center(crop):
            i, j, end_row, end_col = crop
            return np.array([(i + end_row) / 2, (j + end_col) / 2])

        crop_centers = np.array([crop_center(crop) for crop in self.valid_indices])

        selected_indices = [random.randint(0, len(crop_centers) - 1)]

        for _ in range(1, crops_to_generate):
            remaining_indices = [i for i in range(len(crop_centers)) if i not in selected_indices]
            max_dist_idx = max(remaining_indices, key=lambda idx: min(np.linalg.norm(crop_centers[idx] - crop_centers[selected], ord=2) for selected in selected_indices))
            selected_indices.append(max_dist_idx)

        selected_crops = [self.valid_indices[i] for i in selected_indices]

        output_dir = "data/true_crops"
        file_prefix = "greenland_crop_"

        if selected_crops:
            print(f"Saving {len(selected_crops)} cropped areas to '{output_dir}'...")
            for i, crop in enumerate(tqdm(selected_crops, desc="Saving crops")):
                file_path = os.path.join(output_dir, f"{file_prefix}{i+1}.npy")
                np.save(file_path, np.array(crop)) 
                if i == crops_to_generate - 1: 
                    break
            print("Crops saved successfully!")
        else:
            print("No crops to save.")

        return selected_crops
        
    def overlay_crops_on_mask(self, output_path="figures/crops_overlay.png"):
        """Overlays the generated crops on the mask image and saves it."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.mask_tensor.numpy(), cmap='terrain')

        for crop in self.valid_indices:
            start_row, start_col, end_row, end_col = crop
            rect = patches.Rectangle((start_col, start_row), self.crop_size, self.crop_size, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.title("Mask with Cropped Areas")
        plt.savefig(output_path)
        plt.close()
        print(f"Overlay image saved to '{output_path}'")

if __name__ == '__main__':
    bedmachine_path = "data/Bedmachine/BedMachineGreenland-v5.nc"
    crop_generator = CroppedAreaGenerator(bedmachine_path, num_crops=1000) 
    cropped_areas = crop_generator.generate_and_save_crops()
    print(cropped_areas)

    if cropped_areas:
        print(f"Generated {len(cropped_areas)} cropped areas.")
        crop_generator.overlay_crops_on_mask()

