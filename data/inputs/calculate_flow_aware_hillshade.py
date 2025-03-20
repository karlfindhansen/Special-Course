import os
import rioxarray as rxr
import xarray as xr
import numpy as np
import pdemtools as pdt
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from utils import (
    plot_velocity_magnitude, plot_hillshade, get_flow_direction, 
    plot_for_assesment, plot_aspect, plot_flow_aligned_azimuth, 
    plot_offset_aligned_azimuth, plot_final_result, plot_compare_to_bedmachine
)

def configure_plotting():
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['font.sans-serif'] = "Arial"

def load_velocity_data(vf_path):
    vf = xr.open_dataset(vf_path)
    vfx, vfy = vf['land_ice_surface_easting_velocity'], vf['land_ice_surface_northing_velocity']
    crs = "EPSG:3413"
    vfx.rio.write_crs(crs, inplace=True)
    vfy.rio.write_crs(crs, inplace=True)
    return vfx, vfy

def save_velocity_rasters(vfx, vfy, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vfx.rio.to_raster(os.path.join(output_dir, "vfx.tif"))
    vfy.rio.to_raster(os.path.join(output_dir, "vfy.tif"))

def compute_velocity_magnitude(vx_path, vy_path, show):
    vx = rxr.open_rasterio(vx_path).compute().squeeze()
    vy = rxr.open_rasterio(vy_path).compute().squeeze()
    vv = np.sqrt(vx ** 2 + vy ** 2)
    if show:
        plot_velocity_magnitude(vv, vx, vy)
    return vx, vy, vv

def load_and_preprocess_dem(dem_path, reference_raster):
    dem = rxr.open_rasterio(dem_path)
    dem.rio.write_crs("EPSG:3413", inplace=True)
    dem = dem.rio.reproject_match(reference_raster)
    dem = dem.where(dem >= 0).squeeze()
    return dem

def compute_hillshade(dem, show):
    hillshade = dem.pdt.terrain('hillshade', hillshade_z_factor=2, hillshade_multidirectional=True)
    if show:
        plot_hillshade(dem, hillshade)

def compute_weighted_azimuth(vv, az_speed, az_elev, speed_az_decay=100, speed_uncert_rel_decay=0.25):
    wt_az_elev = np.exp(-vv.values.squeeze() / speed_az_decay)
    wt_az_elev[wt_az_elev > 1] = 1
    wt_az_speed = 1 - wt_az_elev
    wt_az_elev[np.isnan(az_speed)] = 1
    wt_az_speed[np.isnan(az_elev)] = 1
    
    az_mean = np.arctan2(
        (np.sin(az_speed) * wt_az_speed) + (np.sin(az_elev) * wt_az_elev),
        (np.cos(az_speed) * wt_az_speed) + (np.cos(az_elev) * wt_az_elev)
    )
    
    azimuth_flt = (np.rad2deg(az_mean) + 360) % 360
    return azimuth_flt

def save_rasters(output_dir, output_fname, vx, vy, dem, fa_hillshade):
    os.makedirs(output_dir, exist_ok=True)
    vx.rio.to_raster(os.path.join(output_dir, f'{output_fname}_vx.tif'), compress='ZSTD', predictor=3, zlevel=1)
    vy.rio.to_raster(os.path.join(output_dir, f'{output_fname}_vy.tif'), compress='ZSTD', predictor=3, zlevel=1)
    dem.rio.to_raster(os.path.join(output_dir, f'{output_fname}_dem.tif'), compress='ZSTD', predictor=3, zlevel=1)
    fa_hillshade.rio.to_raster(os.path.join(output_dir, f'{output_fname}_flowalignedhillshade.tif'), compress='ZSTD', predictor=3, zlevel=1)

def main():
    configure_plotting()
    show = False

    vfx, vfy = load_velocity_data("data/inputs/Ice_velocity/Promice_AVG5year.nc")
    save_velocity_rasters(vfx, vfy, "data/inputs/hillshade")
    vx, vy, vv = compute_velocity_magnitude("data/inputs/hillshade/vfx.tif", "data/inputs/hillshade/vfy.tif", show)
    
    dem = load_and_preprocess_dem("data/inputs/arcticdem_extracted/arcticdem_mosaic_500m_v4.1_dem.tif", vx)
    compute_hillshade(dem, show)
    
    flow_direction = get_flow_direction(vx, vy)
    if show:
        plot_for_assesment(flow_direction, vx=vx, vy=vy)
    
    aspect = dem.pdt.terrain('aspect')
    if show:
        plot_aspect(aspect)
    
    az_speed = np.deg2rad(flow_direction.values).squeeze()
    az_elev = np.deg2rad(aspect.values).squeeze()
    azimuth_flt = compute_weighted_azimuth(vv, az_speed, az_elev)

    if show:
        plot_flow_aligned_azimuth(azimuth_flt)
    
    final_azimuth = (azimuth_flt + 90) % 360

    if show:
        plot_offset_aligned_azimuth(final_azimuth)
    
    fa_hillshade = dem.pdt.terrain(
        'hillshade', hillshade_z_factor=100, hillshade_altitude=60,
        hillshade_azimuth=final_azimuth, hillshade_multidirectional=False
    )
    if show:
        plot_final_result(fa_hillshade)

    bedmachine_data = xr.open_dataset("data/inputs/Bedmachine/BedMachineGreenland-v5.nc")
    bedmachine_data.rio.write_crs("EPSG:3413", inplace=True)
    bed = bedmachine_data['bed'].rio.reproject_match(dem, resampling=Resampling.bilinear)
    if show:
        plot_compare_to_bedmachine(fa_hillshade, bed)
    
    save_rasters("data/inputs/hillshade", "macgregortest", vx, vy, dem, fa_hillshade)
    
if __name__ == "__main__":
    main()
