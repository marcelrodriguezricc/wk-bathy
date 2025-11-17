# Compile a list of dates and times where significant wave height greater than 1M

# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import xarray as xr
from utils.data_classes import AOI, params
from utils.functions import iterate_offset
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# User Parameters
num_days = params.num_days

# Set directory for loading datasets
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
dir = ROOT_DIR / "data" / "models"

# For each AOI...
for a in aoi_list:

    # Establish center date based on CRM creation date
    center_date = datetime.fromisoformat(str(a.crm_date))

    # Initialize arrays for storing dates and significant wave heights greater than 1
    dates_gt1 = []
    swh_gt1 = []

    # Set prefix for saving dataset
    header = Path(a.filename).stem

    # For each day "num_days" before and after the center date...
    for offset in iterate_offset(num_days):

        # Get day based on current offset
        d = (center_date + timedelta(days=offset) + timedelta(days=a.date_offset)).date()

        # Get path to wave model for day and load
        path = dir / f"{header}_wave_{d}.nc"

        # Load dataset, get significant wave height, and calculate daily mean
        ds = xr.open_dataset(path)
        swh = ds["VHM0"]
        swh_avg = float(swh.mean().values)

        # If daily mean is greater than 1...
        if swh_avg > 1.0:
           
           # Convert datetime to string
           date_str = d.strftime("%Y-%m-%d")

           # Store date and SWH in respective arrays
           dates_gt1.append(date_str)
           swh_gt1.append(swh_avg)
    
    # Store arrays in AOI object
    a.swh_array = swh_gt1
    a.swh_dates = dates_gt1

# Prepare for JSON
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")

