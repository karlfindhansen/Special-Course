import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import torch

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

def create_mask(ice_velocity, mass_balance_tensor, lat_deg: int, lat_min: float, lat_hem: str,
                lon_deg: int, lon_min: float, lon_hem: str, area_around_point: int):
    """
    Creates a mask for valid data points around a given coordinate.
    If no valid blocks are found at the coordinate, finds the closest valid blocks.
    """
    x_vals = ice_velocity.x.values
    y_vals = ice_velocity.y.values

    x, y = dms_to_epsg3413(lat_deg, lat_min, lat_hem,
                          lon_deg, lon_min, lon_hem)

    x_idx = np.searchsorted(x_vals, x)
    y_idx = np.searchsorted(y_vals[::-1], y)
    y_idx = len(y_vals) - 1 - y_idx  # Flip due to descending Y

    # === Create tensors ===
    ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)
    
    def is_valid_point(y, x):
        """Check if a point is valid in both datasets"""
        if (0 <= y < ice_velocity_tensor.shape[0] and 
            0 <= x < ice_velocity_tensor.shape[1]):
            return (not torch.isnan(ice_velocity_tensor[y, x]) and 
                   not torch.isnan(mass_balance_tensor[y, x]))
        return False

    def find_largest_square(center_y, center_x, max_radius):
        """Find the largest valid square centered around a point"""
        best_size = 0
        best_y = center_y
        best_x = center_x

        # Try different square sizes
        for size in range(1, max_radius * 2, 2):  # Odd sizes to keep center
            half = size // 2
            valid = True
            
            # Check if all points in this square are valid
            for y in range(center_y - half, center_y + half + 1):
                for x in range(center_x - half, center_x + half + 1):
                    if not is_valid_point(y, x):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_size = size
                best_y = center_y - half
                best_x = center_x - half
            else:
                break

        return best_size, best_y, best_x

    def find_closest_valid_square():
        """Find the closest point with a valid square"""
        max_search_radius = 100
        best_distance = float('inf')
        best_result = None

        for radius in range(1, max_search_radius):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) == radius or abs(dx) == radius:  # Only check the perimeter
                        new_y = y_idx + dy
                        new_x = x_idx + dx
                        
                        if is_valid_point(new_y, new_x):
                            size, start_y, start_x = find_largest_square(new_y, new_x, area_around_point)
                            if size >= 11:  # Minimum size needed for blocks
                                distance = np.sqrt(dy**2 + dx**2)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_result = (size, start_y, start_x)
                                    print(f"Found valid square at distance {distance:.2f}")
                                    return best_result  # Return first valid square found

        return best_result

    # Try to find square at original coordinate
    size, start_y, start_x = find_largest_square(y_idx, x_idx, area_around_point)
    
    # If no valid square found, search for closest one
    if size < 11:  # Need at least size 11 for blocks
        print("No valid square at specified coordinate, searching nearby...")
        result = find_closest_valid_square()
        if result is not None:
            size, start_y, start_x = result
        else:
            print("No valid squares found in search area")
            return None, []

    # Create the mask
    mask = torch.zeros_like(ice_velocity_tensor, dtype=torch.bool)
    if size > 0:
        for i in range(size):
            for j in range(size):
                mask[start_y + i, start_x + j] = True
        
        print(f"Found valid square with side length: {size}")
        print(f"Starting at coordinates: ({start_y}, {start_x})")
    
    # Plot mask overlayed on mass balance
    plt.figure(figsize=(10, 8))
    plt.imshow(mass_balance_tensor.numpy(), cmap='viridis', origin='lower')
    plt.imshow(mask.numpy(), alpha=0.5, cmap='gray')
    #plt.plot(x_idx, y_idx, 'r*', markersize=10, label='Requested coordinate')
    plt.title('Mask Overlayed on Mass Balance')
    plt.colorbar(label='Mass Balance')
    plt.legend()
    plt.savefig("figures/mask_overlayed_on_mass_balance.png", dpi=300, bbox_inches='tight')
    plt.close()

    block_size = 11
    blocks = []
    coords = []

    # Find valid square blocks
    for i in range(start_y, start_y + size - block_size + 1):
        for j in range(start_x, start_x + size - block_size + 1):
            block = mask[i:i+block_size, j:j+block_size]
            if block.all():
                blocks.append(block)
                coords.append((i, j))

    print(f"Found {len(blocks)} blocks of size {block_size}x{block_size} within the valid square")

    return mask, coords

if __name__ == '__main__':

    glaciers = {
        'Kankalusat' : {'lat_deg': 68, 'lat_min': 38, 'lat_hem': 'N',
                        'lon_deg': 33, 'lon_min': 0, 'lon_hem': 'W'},
        'Siorallip Sermia': {'lat_deg': 60, 'lat_min': 51, 'lat_hem': 'N',
                         'lon_deg': 44, 'lon_min': 52, 'lon_hem': 'W'},
        'Sermeq Kangilleq': {'lat_deg': 60, 'lat_min': 58, 'lat_hem': 'N',
                            'lon_deg': 44, 'lon_min': 55, 'lon_hem': 'W'},
        'Sermilik': {'lat_deg': 60, 'lat_min': 59, 'lat_hem': 'N',
                    'lon_deg': 46, 'lon_min': 59, 'lon_hem': 'W'},
        'Naajat Sermiat': {'lat_deg': 61, 'lat_min': 2, 'lat_hem': 'N',
                        'lon_deg': 46, 'lon_min': 35, 'lon_hem': 'W'},
        'Eqalorutsit Killiit Sermiat': {'lat_deg': 61, 'lat_min': 17, 'lat_hem': 'N',
                                        'lon_deg': 46, 'lon_min': 9, 'lon_hem': 'W'},
        'Qooqqup Sermia': {'lat_deg': 61, 'lat_min': 20, 'lat_hem': 'N',
                        'lon_deg': 44, 'lon_min': 59, 'lon_hem': 'W'},
        'Eqalorutsit Kangilliit Sermiat': {'lat_deg': 61, 'lat_min': 22, 'lat_hem': 'N',
                                        'lon_deg': 45, 'lon_min': 44, 'lon_hem': 'W'},
        'Sermiligaarsuup Sermia': {'lat_deg': 61, 'lat_min': 38, 'lat_hem': 'N',
                                'lon_deg': 48, 'lon_min': 8, 'lon_hem': 'W'},
        'Uukkaasorsuaq': {'lat_deg': 61, 'lat_min': 58, 'lat_hem': 'N',
                        'lon_deg': 48, 'lon_min': 38, 'lon_hem': 'W'},
        'Sioqqap Sermia': {'lat_deg': 62, 'lat_min': 33, 'lat_hem': 'N',
                        'lon_deg': 49, 'lon_min': 53, 'lon_hem': 'W'},               
    }

    dms_to_epsg3413(glaciers['Kankalusat']['lat_deg'],
                    glaciers['Kankalusat']['lat_min'], glaciers['Kankalusat']['lat_hem'],
                    glaciers['Kankalusat']['lon_deg'], glaciers['Kankalusat']['lon_min'], glaciers['Kankalusat']['lon_hem'])
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

    create_mask(ice_velocity_data['land_ice_surface_easting_velocity'],mass_balance, 68, 33, 'N', 33, 0, 'W', 100)