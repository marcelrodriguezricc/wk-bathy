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

    # Get dates list from AOI object
    dates = a.optical_dates
    
    # Get bounding box min & max latitude & longitude for dataset query
    min_lon = float(a.lon) - float(a.bbox_lon)
    min_lat = float(a.lat) - float(a.bbox_lat)
    max_lon = float(a.lon) + float(a.bbox_lon)
    max_lat = float(a.lat) + float(a.bbox_lat)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # For each date in list...
    for d in dates:

        # Establish path and append with filename
        path = data_dir / f"{a.filename}_optical_{d}.tif" # **** NEEDS TO BE CHANGED ****

        # Load dataset
        vis = rxr.open_rasterio(path)
        vis4326 = vis.rio.reproject(4326)
        vis_clip = vis4326.rio.clip_box(*bbox) 

        # Reorder to (y, x, band) and scale to 0 – 1
        arr = vis_clip.transpose("y","x","band").values.astype("float32") / 255.0

        # Initialize figure and set parameters
        fig = plt.figure()
        plt.imshow(np.clip(arr, 0, 1))
        plt.title(f"{a.name}, {d}, Sentinel-2 Imagery")
        plt.axis("off")
        
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
        