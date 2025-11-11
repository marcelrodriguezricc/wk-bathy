# Download wave model based on bounding box and window around CRM creation date

# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from datetime import datetime, timedelta
from pathlib import Path
from copernicusmarine import subset
from copernicusmarine import get

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# User Parameters
max_days = 60

# For each AOI...
for a in aoi_list:
    
    # Fetch center date
    center_date = datetime.fromisoformat(str(a.crm_date))

    # If center data is less than earliest extent of Wave Forecast & Analysis Dataset, use Waves Reanalysis Dataset
    if center_date > datetime(2022, 10, 31):
        ds_name = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
    else:
        ds_name = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

    # Set output folder where plot will be saved & file name
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent
    outdir = ROOT_DIR / "data" / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    fname_stem = Path(a.filename).stem
    outpath = outdir / f"{fname_stem}_waves_subset.nc"

    # Specify subset to download
    subset(
        dataset_id = ds_name,
        variables = ["VHM0"], # Significant wave height
        minimum_longitude = a.lon - a.bbox_lon,
        maximum_longitude= a.lon + a.bbox_lon,
        minimum_latitude = a.lat - a.bbox_lat,
        maximum_latitude = a.lat + a.bbox_lat,
        start_datetime = center_date + timedelta(days=max_days),
        end_datetime = center_date - timedelta(days=max_days),
        output_filename = outpath,
    )

