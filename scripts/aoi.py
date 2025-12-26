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

ns_lat = dms_to_decimal(degrees = 21, minutes =  41, seconds = 22, direction = "N") # 21°41'22"N 
ns_lon = dms_to_decimal(degrees = 158, minutes =  6, seconds = 8, direction = "W") # 158°06'08"W
aoi_1 = AOI(name = "North Shore, O'ahu, Hawaii",
            filename = "ns",
            lat = ns_lat,
            lon = ns_lon,
            bbox_lat = 0.1,
            bbox_lon = 0.1,
            date_offset = 0,
            buoy_number = 51201,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol10_2023.nc",
            )

# Golden Gate, California
gg_lat = dms_to_decimal(degrees = 37, minutes =  45, seconds = 33, direction = "N") # 37°45'33"N
gg_lon = dms_to_decimal(degrees = 122, minutes =  39, seconds = 57, direction = "W") # 122°39'57"W
aoi_2 = AOI(name = "Golden Gate, California",
            filename = "gg",
            lat = gg_lat,
            lon = gg_lon,
            bbox_lat = 0.2,
            bbox_lon = 0.2,
            date_offset = 0,
            buoy_number = 46237,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol7_2024.nc",
            )

# Wassabo Beach, Florida
wb_lat = dms_to_decimal(degrees = 27, minutes =  43, seconds = 31, direction = "N") # 27°43'31"N
wb_lon = dms_to_decimal(degrees = 80, minutes =  15, seconds = 6, direction = "W") # 80°15'06"W
aoi_3 = AOI(name = "Wassabo Beach, Florida",
            filename = "wb",
            lat = wb_lat,
            lon = wb_lon,
            bbox_lat = 0.2,
            bbox_lon = 0.2,
            date_offset = 0,
            buoy_number= 41114,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol3_2023.nc",
            )


# Rincon, Puerto Rico
pr_lat = dms_to_decimal(degrees = 18, minutes =  23, seconds = 28, direction = "N") # 18°23'28"N
pr_lon = dms_to_decimal(degrees = 67, minutes =  17, seconds = 36, direction = "W") # 67°17'36"W
aoi_4 = AOI(name = "Rincon, Puerto Rico", 
            filename = "pr",
            lat = pr_lat, 
            lon = pr_lon,
            bbox_lat = 0.1,
            bbox_lon = 0.1,
            date_offset = 150,
            buoy_number= 41115,
            crm_link = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/cudem/crm_vol9_2023.nc",
            )

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
