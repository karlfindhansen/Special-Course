# i want to open a .grip file
import os
import cfgrib

grib_file_path = os.path.join("data", "Snow_acc", "b3c4ca8dbccd70c683bc08a3d9f1d684.grib")

ds = cfgrib.open_dataset(grib_file_path)

#print(ds)

# sf: snowfall
# smlt: snowmelt
# es: snow_evaporation

print(ds.sf.values)

# plot the snowfall data
import matplotlib.pyplot as plt

plt.imshow(ds.sf.values)
plt.colorbar()
plt.show()