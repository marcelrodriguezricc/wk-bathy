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
img_outdir = ROOT_DIR / "images" / "windowed"
img_outdir.mkdir(parents=True, exist_ok=True)

window_size_m = 2000.0
overlap = 0.9

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

    # Determine number of pixels for window dimensions
    win_nx = int(window_size_m / abs(dx))
    win_ny = int(window_size_m / abs(dy))   

    # Make number odd so there is a clear center
    if win_nx % 2 == 0:
        win_nx += 1
    if win_ny % 2 == 0:
        win_ny += 1

    # Calculate the amount the center will shift from one window to the next
    step_x = max(1, int(win_nx * (1 - overlap)))
    step_y = max(1, int(win_ny * (1 - overlap)))

    # Open SAR image raster
    with rasterio.open(data_path) as src:

        # Assign data to variable
        sar_arr = src.read(1)

        # Initialize window center array
        window_centers = []

        # Iterate through each window
        for i, (window, cx, cy, j0, i0) in enumerate(
            iter_windows(sar_arr, lat, lon,
                win_nx, win_ny, step_x, step_y,
                max_nan_fraction = 0.3
            )
        ):
            window_centers.append((i0, j0))

        # Initialize plot
        fig, ax = plt.subplots(figsize = (8, 8))
        im = ax.imshow(sar_arr, cmap="gray", origin="upper")
        ax.set_title(f"{a.name}, Windowed, {date}")

        # Add a red rectangle for each window
        half_wx = win_nx // 2
        half_wy = win_ny // 2

        for (i_pix, j_pix) in window_centers:
            # Top-left corner of the window in pixel coordinates
            x0 = i_pix - half_wx
            y0 = j_pix - half_wy

            rect = mpatches.Rectangle(
                (x0, y0),          # (x, y) of lower-left corner
                win_nx,            # width
                win_ny,            # height
                fill=False,
                edgecolor="red",
                linewidth=0.5
            )
            ax.add_patch(rect)
        
        # No axes
        ax.axis('off')

        # Save plot
        img_outpath = img_outdir / f"{a.filename}_windowed_{date}.png"
        if img_outpath.exists():
            print(f"Window plot already exists for {a.name}, skipping save: {img_outpath}")
        else:
            plt.savefig(img_outpath, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saved window plot for {a.name} → {img_outpath}")
