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
import gzip
import io
import requests
import numpy as np
import pandas as pd
import xarray as xr

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
outdir = ROOT_DIR / "data" / "swell"
outdir.mkdir(parents=True, exist_ok=True)
buoy_dir = ROOT_DIR / "data" / "buoy"
buoy_dir.mkdir(parents=True, exist_ok=True)

# NDBC column layout
ndbc_cols = [
    "YY", "MM", "DD", "hh", "mm",
    "WDIR", "WSPD", "GST",
    "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP",
    "VIS", "TIDE",
]

# For each AOI...
for a in aoi_list:

    # Fetch center date
    center_date = datetime.fromisoformat(str(a.crm_date))

    # Get year
    year = center_date.year
    
    # Get buoy number
    buoy_id = a.buoy_number

    # Load buoy file once per AOI
    buoy_path = buoy_dir / f"{a.filename}_buoy_{year}.txt.gz"

    # If file doesn't already exist, download from NOAA NDBC, else skip
    if not buoy_path.exists():
        url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/{buoy_id}h{year}.txt.gz"
        print(f"Downloading → {url}")
        r = requests.get(url)
        if r.status_code != 200:
            print(f"  ERROR: Could not download {url}, skipping AOI.")
            continue
        buoy_path.write_bytes(r.content)
    else:
        print(f"Buoy textfile already exists for {a.name}, skipping save: {buoy_path}")

    # Read buoy data
    df = pd.read_csv(
        buoy_path,
        sep='\s+',
        comment="#",
        header=None,
        names=ndbc_cols,
        na_values=[99.0, 99.00, 999.0, 9999.0],
    )

    # For each day "num_days" before and after the center date...
    for offset in iterate_offset(num_days):

        # Get day based on current offset
        d = (center_date + timedelta(days=offset) + timedelta(days=a.date_offset)).date()

        # Set filename for saving dataset locally
        outpath = outdir / f"{a.filename}_wave_{d}.nc"

        # If file already exists, skip
        if outpath.exists():
            print(f"Skipping {outpath} (already exists)")
            continue
        
        # Filter buoy rows for this date
        mask = (
            (df["YY"] == d.year) &
            (df["MM"] == d.month) &
            (df["DD"] == d.day)
        )
        day_df = df.loc[mask]

        if day_df.empty:
            print(f"No buoy data for {d} at {buoy_id}, {a.name}, skipping.")
            continue

        # Extract WVHT (Hs), DPD (Tp-like), and MWD (direction)
        hs = day_df["WVHT"].astype(float)
        dpd = day_df["DPD"].astype(float)
        mwd = day_df["MWD"].astype(float)

        # Clean up missing / sentinel values
        hs = hs[hs.notna()]
        dpd = dpd[dpd.notna() & (dpd < 90.0)]     # filter out 99.0 etc.
        mwd = mwd[mwd.notna()]

        if hs.empty or dpd.empty or mwd.empty:
            print(f"  No valid WVHT/DPD/MWD for {d} at {buoy_id}, skipping.")
            continue

        # Daily means
        vhm0 = float(hs.mean())
        vtpk = float(dpd.mean())
        vmdr = float(mwd.mean())

        time_coord = np.array([np.datetime64(d)])
        ds = xr.Dataset(
            data_vars={
                "VHM0": (["time"], [vhm0]),
                "VTPK": (["time"], [vtpk]),
                "VMDR": (["time"], [vmdr]),
            },
            coords={
                "time": time_coord,
            },
            attrs={
                "source": f"NOAA NDBC buoy {buoy_id} daily mean",
            },
        )

        ds.to_netcdf(outpath, mode="w")
        ds.close()
    
# Prepare for JSON (unchanged)
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")