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

def create_mask(ice_velocity, lat_deg: int, lat_min: float, lat_hem: str,
                    lon_deg: int, lon_min: float, lon_hem: str, area_around_point: int):
    
    x_vals = ice_velocity.x.values
    y_vals = ice_velocity.y.values

    x, y = dms_to_epsg3413(lat_deg, lat_min, lat_hem,
                                     lon_deg, lon_min, lon_hem)

    x_idx = np.searchsorted(x_vals, x)
    y_idx = np.searchsorted(y_vals[::-1], y)
    y_idx = len(y_vals) - 1 - y_idx  # Flip due to descending Y

    # === Create tensors ===
    ice_velocity_tensor = torch.tensor(ice_velocity.values.astype(np.float32)).squeeze(0)
    mask = torch.zeros_like(ice_velocity_tensor, dtype=torch.bool)

    mask[y_idx, x_idx] = True

    for i in range(-area_around_point, area_around_point + 1):
        for j in range(-area_around_point, area_around_point + 1):
            if 0 <= y_idx + i < ice_velocity_tensor.shape[0] and 0 <= x_idx + j < ice_velocity_tensor.shape[1]:
                mask[y_idx + i, x_idx + j] = True

    return mask

if __name__ == '__main__':
    dms_to_epsg3413(38, 33, 'N', 33, 0, 'W')