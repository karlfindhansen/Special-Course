import xarray as xr
import torch
import numpy as np
import random
from tqdm import tqdm

# TODO: This file should be a class that finds 5 100 x 100 areas to train in.

class CroppedAreaGenerator:
    def __init__(self, bedmachine_path, crop_size=100, num_crops=None):
        self.bedmachine_path = bedmachine_path
        self.crop_size = crop_size
        self.num_crops = num_crops  # If None, generate as many as possible
        self.bed_tensor, self.mask_tensor = self._load_and_preprocess_data()
        self.valid_indices = self._find_valid_crop_indices()

    def _load_and_preprocess_data(self):
        bedmachine_data = xr.open_dataset(self.bedmachine_path)
        errbed_tensor = torch.tensor(bedmachine_data['errbed'].values.astype(np.float32))
        bed_tensor = torch.tensor(bedmachine_data['bed'].values.astype(np.float32))

        errbed_tensor = torch.flip(torch.rot90(errbed_tensor, 2, dims=(0, 1)), dims=(1,))
        bed_tensor = torch.flip(torch.rot90(bed_tensor, 2, dims=(0, 1)), dims=(1,))

        mask_tensor = (errbed_tensor < 5).float()
        return bed_tensor, mask_tensor

    def _find_valid_crop_indices(self):
        valid_indices = []
        h, w = self.mask_tensor.shape
        for i in tqdm(range(h - self.crop_size + 1), desc="Finding crops"):
            for j in range(w - self.crop_size + 1):
                crop_mask = self.mask_tensor[i:i + self.crop_size, j:j + self.crop_size]
                if torch.all(crop_mask == 1):  # Check if ALL mask values in the crop are 1
                    valid_indices.append((i, j))
                    print("Found a crop!")
        return valid_indices

    def generate_crops(self):
        if self.num_crops is None:
            crops_to_generate = len(self.valid_indices)  # Generate all possible
        else:
            crops_to_generate = min(self.num_crops, len(self.valid_indices))  # Limit if num_crops is specified

        if crops_to_generate == 0:
            return []  # No valid crops found

        selected_indices = random.sample(self.valid_indices, crops_to_generate)  # Randomly select crops
        crops = []
        for i, j in selected_indices:
            crop = self.bed_tensor[i:i + self.crop_size, j:j + self.crop_size]
            crops.append(crop)
        return crops

# Example usage:
if __name__ == '__main__':
    bedmachine_path = "data/Bedmachine/BedMachineGreenland-v5.nc"  # Replace with your path
    crop_generator = CroppedAreaGenerator(bedmachine_path, num_crops=5) # Generate 100 crops, or as many as possible if num_crops=None
    cropped_areas = crop_generator.generate_crops()

    if cropped_areas:
        print(f"Generated {len(cropped_areas)} cropped areas.")
    for i, crop in enumerate(cropped_areas):
        print(f"Crop {i+1} shape: {crop.shape}") # Should be torch.Size([100, 100])
        # ... use the crops for training ...
    else:
        print("No valid crops found.")

    # To get a single crop:
    crop_generator_single = CroppedAreaGenerator(bedmachine_path, num_crops=1)
    single_crop = crop_generator_single.generate_crops()
    if single_crop:
        print(single_crop[0].shape) #Should be torch.Size([100, 100])




