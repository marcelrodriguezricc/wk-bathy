# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import requests
from utils.data_classes import AOI
from utils.functions import normalize_href
from pystac_client import Client
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Set directory and prefix for saving dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
data_dir = ROOT_DIR / "data" / "st1"
data_dir.mkdir(parents=True, exist_ok=True)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Link to earth-search catalog
catalog = Client.open("https://earth-search.aws.element84.com/v1")

for a in aoi_list:

    # Get dates from data for which SWH was > 1m
    swh_dates = list(a.data.keys())

    # Get bounding box min & max latitude & longitude for dataset query
    min_lon = float(a.lon) - float(a.bbox_lon)
    min_lat = float(a.lat) - float(a.bbox_lat)
    max_lon = float(a.lon) + float(a.bbox_lon)
    max_lat = float(a.lat) + float(a.bbox_lat)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # Initialize array to store dates with imagery and date/item storage
    found_dates = []
    items_list = []

    # For each date with a mean significant wave height > 1m...
    for date in swh_dates:

        # Get date and put in datetime format, remove timestamp
        d = datetime.fromisoformat(date).date()

        # Set start and end of day as a strings
        start_of_day = d.strftime("%Y-%m-%d")
        datetime_range = f"{start_of_day}/{start_of_day}"

        # Query earth-search catalog based on criteria
        items = list(catalog.search(
                collections=["sentinel-1-grd"],
                bbox=bbox,
                datetime=datetime_range, 
                query={
                    "sar:instrument_mode": {"eq": "IW"},
                },
                limit=50
            ).items())
        
        # If no items are found, skip, or else add to list of items for current AOI
        if not items:
            continue
        else:
            items_list.append((date, items))

    # If no items are found for any days with mean SWH >1 meter, print out a warning
    if not items_list:
        print(f"No items found for {a.name}")

    # Print number of days with imagery available for current AOI
    print(f"\n{a.name} has scenes available for {len(items_list)} days")

    # For each date where imagery was available...
    for date, items in items_list:
        for it in items:

            if "vv" in it.assets:
                asset = it.assets["vv"]
                pol = "vv"
            elif "vh" in it.assets:
                asset = it.assets["vh"]
                pol = "vh"
            elif asset is None:
                print(f"No VV or VH asset for {a.name} on {date}, skipping.")
                continue
                
            # Prepare dataset for saving
            href = normalize_href(asset.href)
            ext = Path(href).suffix
            outpath = data_dir / f"{a.filename}_sar_{date}{ext}"
                    

            # Save file, skip if it already exists
            if outpath.exists():
                print(f"Image already exists, skipping download: {outpath}")
            else:
                print(f"Downloading asset → {outpath}")
                resp = requests.get(href, stream=True)
                resp.raise_for_status()

                with outpath.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"Saved {outpath}")

            found_dates.append(date)
            a.data[date]["sar"] = {
                "item_id": it.id,
                "polarization": pol,
                "orbit_state": it.properties.get("sat:orbit_state"),
            }

    # Print date of found datasets and save to AOI object
    print(f"\n{a.name} scenes downloaded for the following dates: {found_dates}")
    
# Prepare for JSON
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")