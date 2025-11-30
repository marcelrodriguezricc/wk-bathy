# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from utils.functions import ll_dist, iter_windows
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label, gaussian_filter, distance_transform_edt

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
data_dir = ROOT_DIR / "data" / "st1" / "masked"
latlon_dir = ROOT_DIR / "data" / "st1" / "subset"
img_outdir = ROOT_DIR / "images" / "full_scene_fft"
img_outdir.mkdir(parents=True, exist_ok=True)

window_size_m = 2000.0     # 2 km
overlap = 0.5              # 50% overlap

# For each AOI...
for a in aoi_list:
    
    # Get date of SAR image and set path for retrieval based on date
    date = a.selected_dates["sar"]["date"]
    data_path = data_dir / f"{a.filename}_sar_masked_{date}.tiff"

    # ---- GET MEAN SPATIAL CHANGE IN METERS FOR LAT AND LON -----

    # Load interpolated latitude and longitude grid for SAR subset image pixels
    latlon_path = latlon_dir / f"{a.filename}_latlon_subset_{date}.npz"
    latlon = np.load(latlon_path)
    lat = latlon["lat"]
    lon = latlon["lon"]

    # Get dimensionality of latitude and longitude grid
    ny, nx = lat.shape

    # Use central row as reference
    j0 = ny // 2

    # Get latitude and longitude for all pixels except last in the row
    lat_row = lat[j0, :-1]
    lon_row = lon[j0, :-1]

    # Get latitude and longitude for all neighboring pixels
    lat_row_next = lat[j0, 1:]
    lon_row_next = lon[j0, 1:]

    # Find the X distance between neighboring pixels
    dxs = ll_dist(lat_row, lon_row, lat_row_next, lon_row_next)

    # Calculate mean of X distances
    dx = np.nanmean(dxs)

    # Use central column as reference
    i0 = nx // 2

    # Get latitude and longitude for all pixels except last in the column
    lat_col = lat[:-1, i0]
    lon_col = lon[:-1, i0]

    # Get latitude and longitude for all neighboring pixels 
    lat_col_next = lat[1:, i0]
    lon_col_next = lon[1:, i0]

    # Find Y distance between neighboring pixels
    dys = ll_dist(lat_col, lon_col, lat_col_next, lon_col_next)

    # Calculate mean of Y distances
    dy = np.nanmean(dys)

    win_nx = int(window_size_m / abs(dx))
    win_ny = int(window_size_m / abs(dy))   
    if win_nx % 2 == 0:
        win_nx += 1
    if win_ny % 2 == 0:
        win_ny += 1
    step_x = max(1, int(win_nx * (1 - overlap)))
    step_y = max(1, int(win_ny * (1 - overlap)))

    with rasterio.open(data_path) as src:
        sar_arr = src.read(1)
        for i, (window, cx, cy, j0, i0) in enumerate(
            iter_windows(sar_arr, lat, lon,
                win_nx, win_ny, step_x, step_y,
                max_nan_fraction=0.3)):

            plt.figure(figsize=(4, 4))
            plt.imshow(window, cmap='viridis')
            plt.title(f"Window #{i}  |  Center: ({cx:.1f}, {cy:.1f})")
            plt.colorbar(label="sigma0 (dB)") # or appropriate SAR variable
            plt.tight_layout()
            plt.show()
