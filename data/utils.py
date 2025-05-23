import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

LARGEST_GLACIER_AREAS = ['Øvre Frederiksborg Gletsjer', 'Seward Gletsjer', 'Sermersuaq', 'Kangerlussuup Sermersua', 'Gronau Gletsjer',
                 'Døren', 'Victor Madsen Gletsjer', 'Storstrømmen', 'Zachariae Isstrøm']

KANKALUSAT = 'Kangerlussuaq Gletsjer'


def split_coordinates(df):
    
    try:
        lat = df['LAT'].str.replace(',', '.').astype(float)
    except:
        lat = df['LAT']
    df['lat_deg'] = lat.abs().astype(int)
    df['lat_min'] = (lat.abs() % 1 * 60).round(4)
    df['lat_hem'] = np.where(lat >= 0, 'N', 'S')
    
    # For longitude
    try:
        lon = df['LON'].str.replace(',', '.').astype(float)
    except:
        lon = df['LON']
    df['lon_deg'] = lon.abs().astype(int)
    df['lon_min'] = (lon.abs() % 1 * 60).round(4)
    df['lon_hem'] = np.where(lon >= 0, 'E', 'W')
    
    return df

def plot_tensor(tensor, title, filename, cmap='viridis', show=False):
    """ Generic function to plot PyTorch tensors and save the figure. """
    plt.figure(figsize=(10, 8))
    tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        plt.imshow(tensor, cmap=cmap, origin='lower')
    else:
        plt.imshow(tensor.cpu().numpy(), cmap=cmap, origin='lower') 

    plt.colorbar(label=title)
    plt.title(title)
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        
def dms_to_epsg3413(lat_deg: int, lat_min: float, lat_hem: str,
                    lon_deg: int, lon_min: float, lon_hem: str):
    """
    Convert latitude/longitude in degrees and minutes to EPSG:3413 projected coordinates.

    Args:
        lat_deg (int): Degrees part of latitude.
        lat_min (float): Minutes part of latitude.
        lat_hem (str): 'N' or 'S' hemisphere.
        lon_deg (int): De   grees part of longitude.
        lon_min (float): Minutes part of longitude.
        lon_hem (str): 'E' or 'W' hemisphere.

    Returns:
        (float, float): x, y coordinates in EPSG:3413
    """
    # Convert to decimal degrees
    lat = lat_deg + lat_min / 60.0
    if lat_hem.upper() == 'S':
        lat *= -1

    lon = lon_deg + lon_min / 60.0
    if lon_hem.upper() == 'W':
        lon *= -1

    # Transform to EPSG:3413
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def create_mask(ice_velocity, mass_balance_tensor, glacier_name: str=None, area_around_point: int = 500):
    """
    Creates a mask for the largest valid 12x12 square near a given glacier coordinate.
    If the target point is invalid, searches nearby to find the closest valid 12x12 region.
    """
    if glacier_name:
        glacier_names = pd.read_csv("data/inputs/glaciers.csv", encoding='latin1', sep=';')
        glacier = glacier_names[glacier_names['Official name'] == glacier_name]
    else:
        glacier = pd.DataFrame({'LAT': [67.008611],'LON': [-50.689167]})
    glacier = split_coordinates(glacier)
    lat_deg, lat_min, lat_hem = glacier['lat_deg'].values[0], glacier['lat_min'].values[0], glacier['lat_hem'].values[0]
    lon_deg, lon_min, lon_hem = glacier['lon_deg'].values[0], glacier['lon_min'].values[0], glacier['lon_hem'].values[0]

    x_vals = ice_velocity.x.values
    y_vals = ice_velocity.y.values

    x, y = dms_to_epsg3413(lat_deg, lat_min, lat_hem, lon_deg, lon_min, lon_hem)

    x_idx = np.searchsorted(x_vals, x)
    y_idx = len(y_vals) - 1 - np.searchsorted(y_vals[::-1], y)  # Flip Y-axis for descending order

    ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)

    def is_valid_square(top_y, top_x, block_size):
        """Check if a block_size x block_size square is fully valid"""
        if (top_y + block_size > ice_velocity_tensor.shape[0] or 
            top_x + block_size > ice_velocity_tensor.shape[1]):
            return False
        block_ice = ice_velocity_tensor[top_y:top_y + block_size, top_x:top_x + block_size]
        block_mb = mass_balance_tensor[top_y:top_y + block_size, top_x:top_x + block_size]
        return not (torch.isnan(block_ice).any() or torch.isnan(block_mb).any())

    def find_closest_valid_block(center_y, center_x, block_size, max_search_radius):
        best_distance = float('inf')
        best_coords = None

        for radius in range(max_search_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Check perimeter only
                    if abs(dy) != radius and abs(dx) != radius:
                        continue

                    y0 = center_y + dy - block_size // 2
                    x0 = center_x + dx - block_size // 2

                    if is_valid_square(y0, x0, block_size):
                        dist = np.sqrt(dy ** 2 + dx ** 2)
                        if dist < best_distance:
                            best_distance = dist
                            best_coords = (y0, x0)
            if best_coords:
                print(best_coords)
                #exit()
                return best_coords
        return None

    block_size = 176
    coords = find_closest_valid_block(y_idx, x_idx, block_size, area_around_point)

    if coords is None:
        print("No valid 12x12 block found within search radius.")
        return None, []

    start_y, start_x = coords
    mask = torch.zeros_like(ice_velocity_tensor, dtype=torch.bool)
    mask[start_y:start_y + block_size, start_x:start_x + block_size] = True

    coords = []
    for i in range(start_y, start_y+block_size, 22):
        for j in range(start_x, start_x + block_size, 22):
            block = mask[i:i+22, j:j+22]
            if block.all():
                coords.append((i,j))

    # Plot mask
    plt.figure(figsize=(10, 8))
    plt.imshow(mass_balance_tensor.numpy(), cmap='viridis', origin='lower')
    plt.imshow(mask.numpy(), alpha=0.5, cmap='gray')
    #plt.plot(x_idx, y_idx, 'r*', markersize=10, label='Requested coordinate')
    plt.title(f'Valid {block_size}x{block_size} Mask Overlayed on Mass Balance')
    plt.colorbar(label='Mass Balance')
    plt.savefig("figures/mask_overlayed_on_mass_balance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Found valid {block_size}x{block_size} block at ({start_y}, {start_x})")
    return mask, coords


if __name__ == '__main__':

    glacier_names = pd.read_csv("data/inputs/glaciers.csv", encoding='latin1', sep=';')
    glacier_names = glacier_names[glacier_names['Type'] == 'GrIS']
    
    glaciers = split_coordinates(glacier_names)

    import os
    import xarray as xr
    import rioxarray as rio
    ice_velocity_path = os.path.join("data", "inputs", "Ice_velocity", "Promice_AVG5year.nc")
    ice_velocity_data = xr.open_dataset(ice_velocity_path)
    ice_velocity_data.rio.write_crs("EPSG:3413", inplace=True)

    mass_balance_path= os.path.join("data", "inputs", "mass_balance", "combined_mass_balance.tif")
    mass_balance = rio.open_rasterio(mass_balance_path)
    mass_balance.rio.write_crs("EPSG:3413", inplace=True)
    mass_balance = mass_balance.rio.reproject_match(ice_velocity_data['land_ice_surface_easting_velocity'])
    mass_balance = torch.tensor(mass_balance.values.astype(np.float32)).squeeze(0)

    mask, coords = create_mask(
            ice_velocity_data['land_ice_surface_easting_velocity'],
            mass_balance,
            #glacier_name=KANKALUSAT,
            area_around_point=2500
        )
    
    print(len(coords))
    exit()

    masks = {}
    largest_masks = []  # Store all glaciers with their mask sizes

    total_glaciers = len(glaciers)
    tqdm_bar = tqdm(
        list(glaciers.iterrows()),
        total=total_glaciers,
        desc="Processing glaciers",
        unit="glacier",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for idx, glacier in tqdm_bar:
        glacier_name = glacier['New Greenlandic name']
        tqdm_bar.set_description(f"Processing {glacier_name:30}")

        mask, coords = create_mask(
            ice_velocity_data['land_ice_surface_easting_velocity'],
            mass_balance,
            glacier['lat_deg'],
            glacier['lat_min'],
            glacier['lat_hem'],
            glacier['lon_deg'],
            glacier['lon_min'],
            glacier['lon_hem'],
            area_around_point=100
        )

        if mask is not None:
            current_mask_size = mask.sum().item()
            masks[glacier['ID']] = (mask, coords)

            # Store info for sorting later
            largest_masks.append({
                'glacier_id': glacier['ID'],
                'official_name': glacier['Official name'],
                'greenlandic_name': glacier['New Greenlandic name'],
                'mask_size': current_mask_size,
                'num_blocks': len(coords),
                'mask': mask,
                'coords': coords
            })

    top_10 = sorted(largest_masks, key=lambda x: x['mask_size'], reverse=True)[:10]

    print("\n=== Top 10 Glaciers by Mask Size ===")
    for i, glacier_info in enumerate(top_10, start=1):
        print(f"{i}. {glacier_info['official_name']} (ID: {glacier_info['glacier_id']})")
        print(f"   Greenlandic name: {glacier_info['greenlandic_name']}")
        print(f"   Mask size: {glacier_info['mask_size']} pixels")
        print(f"   Valid blocks: {glacier_info['num_blocks']}")
