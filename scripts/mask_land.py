# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from pathlib import Path
from utils.functions import find_data
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.ops import unary_union

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory and prefix for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
shoreline_dir = ROOT_DIR / "data" / "shoreline_vectors"

data = []

# For each AOI...
for a in aoi_list:
    shoreline_path =  shoreline_dir / f"{a.filename}_shoreline_shp"
    opt_dir = ROOT_DIR / "data" / "st2" / "selected"
    sar_dir = ROOT_DIR / "data" / "st1" / "selected"
    opt_path = find_data("OPTICAL", opt_dir, a.name, a.filename)
    sar_path = find_data("SAR", sar_dir, a.name, a.filename)
    data.append(opt_path)
    data.append(sar_path)
    shoreline = gpd.read_file(shoreline_path)

    for ds in data:
        with rasterio.open(ds) as src:
            raster_crs = src.crs
            raster_meta = src.meta.copy()
            print(f"Raster CRS: {raster_crs}")
    
    # shoreline_union = unary_union(shoreline.geometry)
    # buffered = gpd.GeoSeries([shoreline_union], crs=raster_crs).buffer(BUFFER_M)
