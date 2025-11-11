# Download CRMs for each AOI

# Run from  root
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import xarray as xr
from utils.data_classes import AOI
from utils.functions import download_crm
from pathlib import Path
from dataclasses import asdict
from datetime import datetime as dt

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Download CRM for each AOI
for a in aoi_list:
    local_nc = download_crm(a.crm_link, out_dir="data/crm")
    a.crm_local = str(local_nc)
    ds = xr.open_dataset(local_nc)
    ts_val = ds.attrs.get("GDAL_TIFFTAG_DATETIME")
    if ts_val is not None:
        ts_str = str(int(ts_val))
        date_time = dt.strptime(ts_str, "%Y%m%d%H%M%S")
        a.crm_date = str(date_time.date())
    else:
        date_str = ds.GDAL.split("released")[-1].strip().replace("/", "-")
        a.crm_date = date_str

# Prepare for JSON
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")
