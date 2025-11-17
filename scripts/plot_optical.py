# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
from utils.data_classes import AOI
from pathlib import Path

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory and prefix for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
data_dir = ROOT_DIR / "data" / "st2"

# Set output folder for plots
outdir_img = ROOT_DIR / "images" / "st2"
outdir_img.mkdir(parents=True, exist_ok=True)

# For each AOI...
for a in aoi_list:

    # Get dates with optical data from AOI object
    dates = [d for d in a.data if "optical" in a.data[d]]
    
    # Get bounding box min & max latitude & longitude for dataset query
    min_lon = float(a.lon) - float(a.bbox_lon)
    min_lat = float(a.lat) - float(a.bbox_lat)
    max_lon = float(a.lon) + float(a.bbox_lon)
    max_lat = float(a.lat) + float(a.bbox_lat)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # For each date in list...
    for d in dates:

        # Establish path and append with filename
        path = data_dir / f"{a.filename}_optical_{d}.tif"

        # Get data points for date essential to selection process
        swh_val = a.data[d]["swh"]
        period_val = a.data[d]["period"]
        direction_val = a.data[d]["direction"]
        cloud_val = a.data[d]["optical"]["cloud_cover"]
        sun_az_val = a.data[d]["optical"]["sun_azimuth"]
        sun_el_val = a.data[d]["optical"]["sun_elevation"]

        # Load dataset
        vis = rxr.open_rasterio(path)
        vis4326 = vis.rio.reproject(4326)
        vis_clip = vis4326.rio.clip_box(*bbox) 

        # Reorder to (y, x, band) and scale to 0 – 1
        arr = vis_clip.transpose("y","x","band").values.astype("float32") / 255.0

        # Initialize figure and set parameters
        fig = plt.figure()
        plt.imshow(np.clip(arr, 0, 1))
        plt.axis("off")

        # Title, comment box, layout
        plt.title(f"{a.name}, {d}", pad=16)
        fig.text(
            0.85, 0.5,                   
            f"Mean SWH: {swh_val:.2f} m\nPeriod: {period_val:.2f} seconds\nDirection: {direction_val:.2f}°\nCloud Coverage: {cloud_val:.2f}\nSolar Azimuth: {sun_az_val:.2f}°\nSolar Elevation: {sun_el_val:.2f}°",
            va="center", ha="left",
            fontsize=10,
            linespacing=2.0,
            bbox=dict(facecolor="white", edgecolor="white")
        )
        plt.tight_layout()

        # Set filename from AOI object for saving plot and append to path string
        fname_stem = Path(a.filename).stem
        outpath = outdir_img / f"{fname_stem}_optical_{d}.png"

        # Save the figure, unless already in folder
        if outpath.exists():
                print(f"Image already exists, skipping save: {outpath}")  
        else:  
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {outpath}")  

        # Optional show
        # plt.show()
        