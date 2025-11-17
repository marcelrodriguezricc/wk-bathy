# Establish areas of interest, compile into a list, and save.

# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from pathlib import Path
from dataclasses import asdict
from utils.data_classes import AOI
from utils.functions import dms_to_decimal

# North Shore, O'ahu, Hawaii
ehukai_lat = dms_to_decimal(degrees = 21, minutes =  40, seconds = 19, direction = "N") # 21°40'19"N
ehukai_lon = dms_to_decimal(degrees = 158, minutes =  3, seconds = 42, direction = "W") # 158°03'42"W
aoi_1 = AOI(name = "North Shore, O'ahu, Hawaii",
            filename = "ehukai",
            lat = ehukai_lat,
            lon = ehukai_lon,
            bbox_lat = 0.05,
            bbox_lon = 0.05,
            date_offset = 0,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol10_2023.nc")

# Golden Gate, California
gg_lat = dms_to_decimal(degrees = 37, minutes =  47, seconds = 35, direction = "N") # 37°47'35"N 
gg_lon = dms_to_decimal(degrees = 122, minutes =  33, seconds = 19, direction = "W")# 122°33'19"W
aoi_2 = AOI(name = "Golden Gate, California",
            filename = "gg",
            lat = gg_lat,
            lon = gg_lon,
            bbox_lat = 0.05,
            bbox_lon = 0.05,
            date_offset = 0,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol7_2024.nc")

# Diamond Shoals, North Carolina
ds_lat = dms_to_decimal(degrees = 35, minutes =  12, seconds = 15, direction = "N") # 35°12'15"N 
ds_lon = dms_to_decimal(degrees = 75, minutes =  29, seconds = 40, direction = "W") # 75°29'40"W
aoi_3 = AOI(name = "Diamond Shoals, North Carolina",
            filename = "dshoals",
            lat = ds_lat,
            lon = ds_lon,
            bbox_lat = 0.05,
            bbox_lon = 0.05,
            date_offset = 0,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol2_2023.nc")

# Punta Jacinto, Puerto Rico
pj_lat = dms_to_decimal(degrees = 18, minutes =  32, seconds = 16, direction = "N") # 18°32'16"N
pj_lon = dms_to_decimal(degrees = 67, minutes =  4, seconds = 23, direction = "W") # 67°04'23"W
aoi_4 = AOI(name = "Punta Jacinto, Puerto Rico", 
            filename = "pj",
            lat = pj_lat, 
            lon = pj_lon,
            bbox_lat = 0.05,
            bbox_lon = 0.05,
            date_offset = 236,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol9_2023.nc")

# Compile array
aoi_list = [aoi_1, aoi_2, aoi_3, aoi_4]

# Prepare for JSON
payload = []
for a in aoi_list:
    d = asdict(a)
    payload.append(d)

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2))
print(f"Saved {len(payload)} AOIs → {out_path}")
