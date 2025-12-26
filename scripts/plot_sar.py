# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import rasterio 
import numpy as np
from utils.functions import build_radar_coordinates, build_lat_lon_grids
import matplotlib.pyplot as plt
from utils.data_classes import AOI
from pathlib import Path
from rasterio.windows import Window
import cartopy.crs as ccrs

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory and prefix for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
data_dir = ROOT_DIR / "data" / "st1" / "raw"

# Set output folder for plots
outdir_img = ROOT_DIR / "images" / "st1 / selection_plots"
outdir_img.mkdir(parents=True, exist_ok=True)

# For each AOI...
for a in aoi_list:

    # Get dates with optical data from AOI object
    dates = [d for d in a.data if "sar" in a.data[d]]
    
    # Get bounding box min & max latitude & longitude for dataset query
    min_lon = float(a.lon) - float(a.bbox_lon)
    min_lat = float(a.lat) - float(a.bbox_lat)
    max_lon = float(a.lon) + float(a.bbox_lon)
    max_lat = float(a.lat) + float(a.bbox_lat)

    # For each date in list...
    for d in dates:

        # Establish path and append with filename
        path = data_dir / f"{a.filename}_sar_{d}.tiff"
        xml_path = data_dir / f"{a.filename}_sar_{d}.xml"

        # Get data points for date essential to selection process
        swh_val = a.data[d]["swh"]
        period_val = a.data[d]["period"]
        direction_val = a.data[d]["direction"]
        pol = a.data[d]["sar"]["polarization"]
        orb = a.data[d]["sar"]["orbit_state"]

        # Use XML metadata to get slant range, azimuth time for each column/row, latitude, and longitude for each pixel of raw SAR measurement data
        slant_range_m, azimuth_time = build_radar_coordinates(path, xml_path)
        lat_full, lon_full = build_lat_lon_grids(path, xml_path)
        height, width = lat_full.shape

        # Mask for just values within AOI boundary box
        mask = (
            (lon_full >= min_lon) & (lon_full <= max_lon) &
            (lat_full >= min_lat) & (lat_full <= max_lat)
        )
        r_full = np.broadcast_to(slant_range_m, (height, width))
        r_aoi = r_full[mask]

        # Get mean cutoff wavelength for Velocity Brunching mechanism
        mean_r = float(r_aoi.mean())
        velocity_st1 = 7100.0  # m/s
        mean_lmin = mean_r * np.sqrt(swh_val) / velocity_st1

        # Get subset of data based on bounding box mask
        rows, cols = np.where(mask)
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        with rasterio.open(path) as src:
            window = Window.from_slices(
                (row_min, row_max + 1),
                (col_min, col_max + 1)
             )
            backscatter = src.read(1, window=window).astype("float32")

        # Apply bounding box mask to get interpolated latitude and longitudes as well
        lat_subset = lat_full[row_min:row_max+1, col_min:col_max+1]
        lon_subset = lon_full[row_min:row_max+1, col_min:col_max+1]

        # Normalize baskscatter values, convert to decibels for visualization
        bs_norm = backscatter / np.nanmax(backscatter)
        bs_db = 10 * np.log10(np.clip(bs_norm, 1e-6, None))

        # Set projection
        proj = ccrs.PlateCarree(globe=ccrs.Globe(datum='WGS84'))
        fig, ax = plt.subplots(subplot_kw={"projection": proj})

        # Flip horizontal and vertical if satellite is ascending, flip horizontal if descending, for correct image orientation
        if orb == "ascending":
            bs_db = bs_db[::-1, :]
            lat_subset = lat_subset[::-1, :]
            lon_subset = lon_subset[::-1, :]
        elif orb == "descending":
            bs_db = bs_db[:, ::-1]
            lon_subset = lon_subset[::-1, :]
             

        # Plot
        fig = plt.figure()
        plt.imshow(bs_db, cmap="gray", vmin=-25, vmax=0, origin="upper")
        plt.axis("off")
        plt.title(f"{a.name}, {d}, Sentinel-1", pad=16)
        fig.text(
            0.85, 0.5,                   
            f"Mean SWH: {swh_val:.2f} m\nPeriod: {period_val:.2f} seconds\nDirection: {direction_val:.2f}°\nPolarization: {pol}\nOrbit State: {orb}\nMean Cutoff Wavelength: {mean_lmin:.2f} m",
            va="center", ha="left",
            fontsize=10,
            linespacing=2.0,
            bbox=dict(facecolor="white", edgecolor="white")
        )
        plt.tight_layout()
        
        # Establish path for image output based on filename and date
        fname_stem = Path(a.filename).stem
        outpath = outdir_img / f"{fname_stem}_sar_{d}.png"

        # Save the figure, unless already in folder
        if outpath.exists():
                print(f"Image already exists, skipping save: {outpath}")  
        else:  
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {outpath}")  

