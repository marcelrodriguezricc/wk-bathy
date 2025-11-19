# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import requests
from utils.data_classes import AOI
from pystac_client import Client
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

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
                collections=[sentinel-1-grd],
                bbox=bbox,
                datetime=datetime_range, 
                query={"eo:cloud_cover": {"lte": 20}},
                limit=50
            ).items())