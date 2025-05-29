import matplotlib.pyplot as plt
import rioxarray as rio
import numpy as np

local_nc_filename = "data/inputs/Bedmachine/BedMachineGreenland-v5.nc"
mass_balance = "data/inputs/mass_balance/combined_mass_balance.tif"

if __name__ == "__main__":

    #download_file(url_nc_file, local_nc_filename)
    #download_file(url_tif_file, local_tif_filename)

    # inspect the file
    import xarray as xr
    ds = xr.open_dataset(local_nc_filename)
    # plot ds mask

    err_bed = ds['errbed']
    #mask_mask = ds['mask']

    mass_balance = rio.open_rasterio(mass_balance).squeeze()
    mass_balance.rio.write_crs("EPSG:3413", inplace=True)
    err_bed.rio.write_crs("EPSG:3413", inplace=True)

    err_bed = err_bed.rio.reproject_match(mass_balance)
    
    mass_balance_mask = ~np.isnan(mass_balance)
    
    mass_balance_mask = ~np.isnan(mass_balance)
    
    # Apply mask to err_bed and create binary mask
    masked_err_bed = err_bed.where(mass_balance_mask)
    binary_mask = (masked_err_bed > 30).astype(int)

    #fig, ax = plt.subplots(figsize=(15, 10))

    plt.imshow(binary_mask, cmap='binary')
    
    #contours = ax.contour(binary_mask, levels=[0.5], colors='red', linewidths=0.5)
    
    #plt.colorbar(label='Error Bed Mask (0: error ≤ 30, 1: error > 30)')
    plt.axis('off')
    plt.title('Errorbed')
    plt.tight_layout()
    plt.savefig('data/inputs/Bedmachine/error_bed.png',dpi=500)
   #plt.show()
