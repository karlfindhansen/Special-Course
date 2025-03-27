# i want to open a .grip file
import os
import xarray as xr
import cfgrib
import rioxarray
import matplotlib.pyplot as plt

grib_file_path = os.path.join("data", "inputs", "Snow_acc", "b3c4ca8dbccd70c683bc08a3d9f1d684.grib")
ice_velocity_path = os.path.join("data", "inputs", "ice_velocity", "Promice_AVG5year.nc")
save_path = os.path.join("data", "inputs", "Snow_acc", "snow_acc_rate.tif")

ds = cfgrib.open_dataset(grib_file_path)
ice_velocity_data = xr.open_dataset(ice_velocity_path)
ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

# sf: snowfall
# smlt: snowmelt
# es: snow_evaporation

snow_accumulation_rate = ds.sf.values - ds.smlt.values - ds.es.values

snow_accumulation_rate_da = xr.DataArray(
    snow_accumulation_rate,
    dims=["y", "x"],  
    coords={"y": ds.latitude.values, "x": ds.longitude.values}, 
    attrs={"long_name": "Snow Accumulation Rate", "units": "mm/day"}
)

snow_accumulation_rate_da.rio.write_crs("EPSG:4326", inplace=True) 

snow_accumulation_rate_da = snow_accumulation_rate_da.rio.reproject_match(ice_velocity_data)
snow_accumulation_rate_da.rio.to_raster(save_path)
#print(ds.sf.variables)

plt.imshow(snow_accumulation_rate_da.values)
plt.colorbar()
plt.show()