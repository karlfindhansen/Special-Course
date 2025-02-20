import os

from data_preprocessing import ArcticDataloader, crop_data_from_dataloader
from torch.utils.data import DataLoader

dataset = ArcticDataloader(
        bedmachine_path="data/Bedmachine/BedMachineGreenland-v5.nc",
        arcticdem_path="data/Surface_elevation/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/Ice_velocity/Promice_AVG5year.nc",
        snow_accumulation_path = "data/Snow_acc/...",)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

true_crops_folder = "data/true_crops"

if os.listdir(true_crops_folder) == []:
    from find_location import CroppedAreaGenerator
    bedmachine_path = "data/Bedmachine/BedMachineGreenland-v5.nc" 
    crop_generator = CroppedAreaGenerator(bedmachine_path, num_crops=5) 
    cropped_areas = crop_generator.generate_and_save_crops()

cropped_data = crop_data_from_dataloader(dataloader, true_crops_folder)
