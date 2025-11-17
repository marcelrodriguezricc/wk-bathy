# Get CRMs for each AOI, standardize, visualize, and save.

# Run from  root
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import math
import json
from pathlib import Path
from utils.data_classes import AOI
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects
aoi_list = [AOI(**a) for a in aoi_data]
proj = ccrs.PlateCarree()
norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)

# Set directory for saving plot
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
outdir = ROOT_DIR / "images" / "crm_plots"
outdir.mkdir(parents=True, exist_ok=True)
    
# For each AOI...
for a in aoi_list:

    # Establish path and open local CRM
    file_path = str(a.crm_local)
    ds = xr.open_dataset(file_path)

    # Get bounding box extents
    lat_min, lat_max = a.lat - a.bbox_lat, a.lat + a.bbox_lat
    lon_min, lon_max = a.lon - a.bbox_lon, a.lon + a.bbox_lon

    # Clip dataset based on bounding box extents
    ds_clipped = ds["z"].sel(lat = slice(lat_min, lat_max), lon = slice(lon_min, lon_max))

    # Create matplotlib figure & set parameters
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)
    ax.set_title(a.name, pad=12)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="k", alpha=0.3)
    gl.right_labels = gl.top_labels = False
    pcm = ax.pcolormesh(
        ds_clipped["lon"], ds_clipped["lat"], ds_clipped,
        transform=proj,
        norm=norm,
        cmap=(cmocean.cm.topo), 
        shading="auto",
    )
    ax.set_extent([float(lon_min), float(lon_max), float(lat_min), float(lat_max)], crs=proj)
    cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cb.set_label("Elevation (m)")
    
    # Set filename from AOI object for saving plot and append to path string
    fname_stem = Path(a.filename).stem
    outpath = outdir / f"{fname_stem}_crm.png"

    # Save the figure, unless already in folder
    if outpath.exists():
            print(f"Image already exists, skipping save: {outpath}")
    else:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()