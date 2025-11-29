# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from pathlib import Path
from dataclasses import asdict

# Set dates of selected SAR imagery for each AOI
ekuhai_sar_select = "2022-04-28"
gg_sar_select = "2024-12-19"
dshoals_sar_select = "2023-09-12"
pj_sar_select = "2023-02-05"

# Set dates of selected optical imagery for each AOI
ekuhai_opt_select = "2022-04-28"
gg_opt_select = "2024-12-19"
dshoals_opt_select = "2023-08-15"
pj_opt_select = "2023-02-06"

# Prepare dict for saving to JSON
selected_date_config = {
    "North Shore, O'ahu, Hawaii": {
        "sar": ekuhai_sar_select,
        "optical": ekuhai_opt_select,
    },
    "Golden Gate, California": {
        "sar": gg_sar_select,
        "optical": gg_opt_select,
    },
    "Diamond Shoals, North Carolina": {
        "sar": dshoals_sar_select,
        "optical": dshoals_opt_select,
    },
    "Punta Jacinto, Puerto Rico": {
        "sar": pj_sar_select,
        "optical": pj_opt_select,
    },
}

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Prepare for JSON
payload = [asdict(a) for a in aoi_list]

# For each aoi...
for a in aoi_list:

    # Save selected dates to object
    key = a.name
    if key in selected_date_config:
        cfg = selected_date_config[key]
        a.selected_dates["sar"] = {"date": cfg["sar"]}
        a.selected_dates["optical"] = {"date": cfg["optical"]}
        print(f"Selections for S{a.name}: SAR → {cfg['sar']}, Optical → {cfg['optical']}")
    else:
        print(f"Warning: no selected_dates config for AOI name: {a.name}")

# Prepare for JSON *after* modifying AOIs
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")
