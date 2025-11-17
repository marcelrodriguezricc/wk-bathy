# Download wave model subsets based on bounding box and window around CRM creation date

# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI, params
from utils.functions import iterate_offset
from datetime import datetime, time, timedelta
from pathlib import Path
from dataclasses import asdict
from copernicusmarine import subset
from copernicusmarine import get

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# User Parameters
num_days = params.num_days

# Set directory and prefix for saving dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
outdir = ROOT_DIR / "data" / "models"
outdir.mkdir(parents=True, exist_ok=True)

# For each AOI...
for a in aoi_list:
    
    # Fetch center date
    center_date = datetime.fromisoformat(str(a.crm_date))

    # If center data is less than earliest extent of Wave Forecast & Analysis Dataset, use Waves Reanalysis Dataset
    if center_date > datetime(2022, 10, 31):
        ds_name = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
    else:
        ds_name = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

    # Filename from AOI object for saving
    fname_stem = Path(a.filename).stem

    # For each day "num_days" before and after the center date...
    for offset in iterate_offset(num_days):

        # Get day based on current offset
        d = (center_date + timedelta(days=offset) + timedelta(days=a.date_offset)).date()

        # Set filename for saving dataset locally
        outpath = outdir / f"{fname_stem}_wave_{d}.nc"

        # If file already exists, skip
        if outpath.exists():
            print(f"Skipping {outpath} (already exists)")
            continue
        
        # Start of offset date
        start_datetime = datetime.combine(d, time.min)

        # End of offset date
        end_datetime = datetime.combine(d, time.max)

        # Get a subset of wave model data based on variables of interest, bounding box, and date, then save locally
        print(f"Saving subset → {outpath}")
        subset(
            dataset_id = ds_name,
            variables = ["VHM0", "VTPK", "VMDR"],
            minimum_longitude = a.lon - a.bbox_lon,
            maximum_longitude = a.lon + a.bbox_lon,
            minimum_latitude = a.lat - a.bbox_lat,
            maximum_latitude = a.lat + a.bbox_lat,
            start_datetime = start_datetime,
            end_datetime = end_datetime,
            output_filename = outpath,
        )
    
# Prepare for JSON
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")