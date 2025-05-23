import xarray as xr
import rioxarray as rio
import matplotlib.pyplot as plt
import numpy as np


inner_mass_balance_path = 'data/inputs/mass_balance/CCI_GrIS_RA_SEC_5km_Vers3.0_2024-05-31.nc'
outher_mass_balance_path = 'data/inputs/mass_balance/prodem_rel21_dhdt_5y_2019-2023.tif'
ice_velocity_path = 'data/inputs/Ice_velocity/Promice_AVG5year.nc'

inner_mass_balance = xr.open_dataset(inner_mass_balance_path)
inner_mass_balance.rio.write_crs("EPSG:3413", inplace=True)
inner_mass_balance = inner_mass_balance['SEC'] # meters per year
inner_mass_balance = inner_mass_balance.mean(dim='t')
inner_mass_balance.rio.write_crs("EPSG:3413", inplace=True)

outer_mass_balance = rio.open_rasterio(outher_mass_balance_path)
outer_mass_balance.rio.write_crs("EPSG:3413", inplace=True)
outer_mass_balance = outer_mass_balance.where(np.isfinite(outer_mass_balance), np.nan)
outer_mass_balance = outer_mass_balance.isel(band=0)
outer_mass_balance = outer_mass_balance 

ice_velocity_data = xr.open_dataset(ice_velocity_path)
ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)
ice_velocity_data = ice_velocity_data['land_ice_surface_easting_velocity']

inner_mass_balance = inner_mass_balance.rio.reproject_match(ice_velocity_data)
outer_mass_balance = outer_mass_balance.rio.reproject_match(ice_velocity_data)

combined_mass_balance = xr.where(
    np.isfinite(inner_mass_balance) & np.isfinite(outer_mass_balance),
    (inner_mass_balance + outer_mass_balance) / 2,
    xr.where(
        np.isfinite(inner_mass_balance),
        inner_mass_balance, 
        outer_mass_balance  
    )
) * 917

output_path = "data/inputs/mass_balance/combined_mass_balance.tif"
combined_mass_balance.rio.to_raster(output_path)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8)) 

im1 = ax[0].imshow(inner_mass_balance, cmap="viridis")
ax[0].set_title("Inner Mass Balance")
ax[0].axis("off")  
fig.colorbar(im1, ax=ax[0], orientation="vertical", label="Mass Balance (units)")

im2 = ax[1].imshow(outer_mass_balance, cmap="viridis")
ax[1].set_title("Outer Mass Balance")
ax[1].axis("off")  
fig.colorbar(im2, ax=ax[1], orientation="vertical", label="Mass Balance (units)")

im3 = ax[2].imshow(combined_mass_balance, cmap="viridis")
ax[2].set_title("Combined Mass Balance")
ax[2].axis("off")  
fig.colorbar(im3, ax=ax[2], orientation="vertical", label="Mass Balance (units)")

plt.tight_layout()
plt.savefig('data/inputs/mass_balance/combined.png')
plt.show()