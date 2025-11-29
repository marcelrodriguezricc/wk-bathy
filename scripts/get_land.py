# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import os
import requests
import zipfile
from pathlib import Path
from utils.data_classes import AOI

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Set directory and prefix for saving dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
out_dir = ROOT_DIR / "data" / "land_vectors"
out_dir.mkdir(parents=True, exist_ok=True)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# For each AOI...
for a in aoi_list:

    # Get shoreline link from AOI object
    ne_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"

    # Set filename for saving
    outpath = out_dir / "ne_10m_land.zip"

    # If file already exists skip, else download
    if outpath.exists():
        print(f"Zip already exists, skipping download: {outpath}")
    else:
        print(f"Downloading Natural Earth shapefile from { ne_url} ...")
        resp = requests.get(ne_url, stream=True)
        resp.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete, saved to", outpath)

    # Create new folder to save shapefiles
    extract_dir = out_dir / f"ne_10m_shp"
    extract_dir.mkdir(exist_ok=True)

    # If shapefile folder already exists skip, else download
    if extract_dir.exists():
        print(f"Shapefile already exists, skipping extraction: {extract_dir}")
        with zipfile.ZipFile(outpath, "r") as z:
            z.extractall(extract_dir)
    print("Unzipped shapefile(s) to", extract_dir)
