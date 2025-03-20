import matplotlib.pyplot as plt
import numpy as np
import time

SHOW_EVERY = 50

def plot_velocity_magnitude(vv, vx, vy):
    start_time = time.time()
    fig, ax = plt.subplots(layout='constrained')

    vv.plot.imshow(cmap='viridis', ax=ax, cbar_kwargs={'label': 'Velocity [m/a]'})

    show_every = SHOW_EVERY
    skip = slice(None, None, show_every)
    ax.quiver(
        vx.squeeze().x.values[skip], vx.squeeze().y.values[skip], 
        vx.squeeze().values[skip, skip], vy.squeeze().values[skip, skip],
        color='w',
    )

    ax.set_aspect('equal')
    ax.set_title(None)
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"It took {elapsed_time:.2f} seconds to plot velocity magnitude!")

def plot_hillshade(dem, hillshade):
    fig, ax = plt.subplots(layout='constrained')

    dem.plot.imshow(cmap='gist_earth', ax=ax, cbar_kwargs={'label': 'Elevation [m a.s.l.]'})
    hillshade.plot.imshow(ax=ax, cmap='Greys_r', alpha=.3, add_colorbar=False)

    ax.set_aspect('equal')
    ax.set_title(None)
    plt.show()


def get_flow_direction(vx, vy):
    # in degrees clockwise from due north
    
    direction_rad = np.arctan2(vy, vx) 
    direction_deg = np.rad2deg(direction_rad) - 90 # subtract 90 to get from degrees clockwise from east to degrees clockwise from north
    direction_deg = 360 - direction_deg

    return direction_deg
    
def plot_for_assesment(flow_direction, vx, vy):
    fig, ax = plt.subplots(layout='constrained')

    flow_direction.plot.imshow(cmap='twilight', vmin=0, vmax=360, cbar_kwargs={'label': 'Flow direction [˚]'})

    show_every = SHOW_EVERY  # Plot the velocity as a quiver plot, showing only every 50 quivers
    skip = slice(None, None, show_every)
    ax.quiver(vx.squeeze().x.values[skip], vx.squeeze().y.values[skip], vx.squeeze().values[skip, skip], vy.squeeze().values[skip, skip])#, , )

    ax.set_aspect('equal')
    ax.set_title(None)
    plt.show()

def plot_aspect(aspect):
    fig, ax = plt.subplots(layout='constrained')
    aspect.plot.imshow(cmap='twilight', vmin=0, vmax=360, ax=ax, cbar_kwargs={'label': 'Surface aspect [˚]'})
    ax.set_aspect('equal')
    ax.set_title(None)
    plt.show()


def plot_flow_aligned_azimuth(azimuth_flt):
    fig, ax = plt.subplots(layout='constrained')
    azimuth_flt.plot.imshow(cmap='twilight', ax=ax, cbar_kwargs={'label': 'Flow-aligned azimuth [˚]'}, vmin=0, vmax=360, )
    ax.set_aspect('equal')
    ax.set_title(None)
    plt.show()

def plot_offset_aligned_azimuth(final_azimuth):
    fig, ax = plt.subplots(layout='constrained')
    final_azimuth.plot.imshow(cmap='twilight', ax=ax, cbar_kwargs={'label': 'Flow-aligned azimuth, offset 90˚ [˚]'}, vmin=0, vmax=360, )
    ax.set_aspect('equal')
    ax.set_title(None)
    plt.show()

def plot_final_result(fa_hillshade):
    fig, ax = plt.subplots(layout='constrained')

    fa_hillshade.plot.imshow(ax=ax, cmap='Greys_r')#, add_colorbar=False)

    ax.set_aspect('equal')
    ax.set_title('Flow-aware hillshade (MacGregor et al. 2024)')
    plt.show()

def plot_compare_to_bedmachine(fa_hillshade, bed):
    fig, axes = plt.subplots(layout='constrained', ncols=2, figsize=(10,3.7), sharex=True, sharey=True)

    ax = axes[0]
    fa_hillshade.plot.imshow(ax=ax, cmap='Greys_r', add_colorbar=False)
    ax.set_aspect('equal')
    ax.set_title('Flow-aware surface hillshade')

    ax = axes[1]
    bed.plot.imshow(cmap='gist_earth', ax=ax, cbar_kwargs={'label': 'Bed [m/a]'})
    bed_hillshade =  bed.pdt.terrain('hillshade', hillshade_z_factor=2, hillshade_multidirectional=True)
    bed_hillshade.plot.imshow(ax=ax, cmap='Greys_r', alpha=0.6, add_colorbar=False)
    ax.set_aspect('equal')
    ax.set_title('BedMachine bed hillshade')
    plt.show()