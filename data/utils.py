import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygmt as gmt


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
    
def plot_greenland():
    fig = gmt.Figure()
    gmt.makecpt(cmap="jet", series=[-2800, 2800, 200], D="o")
    fig.grdimage(
        region=[-2700000, 2800000, -2200000, 2300000],
        projection="X8c/7c",
       # grid="lowres/bedmap2_bed.tif",
        cmap=True,
        I="+d",
        Q=True,
        # frame='+t"BEDMAP2"'
    )
    fig.show()
        
if __name__ == '__main__':
    plot_greenland()