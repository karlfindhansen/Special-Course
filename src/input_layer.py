from torch.utils.data import DataLoader

from data_preprocessing import ArcticDEMDataset

dataset = ArcticDEMDataset(
        bedmachine_path="data/BedMachineGreenland-v5.nc",
        arcticdem_path="data/arcticdem_mosaic_500m_v4.1.tar",
        ice_velocity_path="data/dataverse_files/Promice_AVG5year.nc"
    )

dataset = DataLoader(dataset, batch_size=1, shuffle=False)

data = next(iter(dataset))

arcticdem = data['arcticdem']
errbed = data['errbed']
ice_velocity_x = data['ice_velocity_x']
ice_velocity_y = data['ice_velocity_y']

print(arcticdem.shape)
print(errbed.shape)
print(ice_velocity_x.shape)
print(ice_velocity_y.shape)
