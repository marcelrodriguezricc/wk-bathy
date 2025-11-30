# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from pathlib import Path
from dataclasses import asdict

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set parameters to tune SAR FFT
ekuhai_sar_per = 85
ekuhai_sar_lmin = 0.6
ekuhai_sar_lmax = 300
gg_sar_per = 95
gg_sar_lmin = 0.95
gg_sar_lmax = 800
dshoals_sar_per = 90
dshoals_sar_lmin = 0.7
dshoals_sar_lmax = 400
pj_sar_per = 80
pj_sar_lmin = 0.7
pj_sar_lmax = 300

# Compile params in dict for iterating
selected_date_config = {
    "North Shore, O'ahu, Hawaii": {
        "fft_per": ekuhai_sar_per,
        "fft_lmin": ekuhai_sar_lmin,
        "fft_lmax": ekuhai_sar_lmax
    },
    "Golden Gate, California": {
        "fft_per": gg_sar_per,
        "fft_lmin": gg_sar_lmin,
        "fft_lmax": gg_sar_lmax
    },
    "Diamond Shoals, North Carolina": {
        "fft_per": dshoals_sar_per,
        "fft_lmin": dshoals_sar_lmin,
        "fft_lmax": dshoals_sar_lmax,
    },
    "Punta Jacinto, Puerto Rico": {
        "fft_per": pj_sar_per,
        "fft_lmin": pj_sar_lmin,
        "fft_lmax": pj_sar_lmax
    },
}

# For each AOI...
for a in aoi_list:
    
    # Save parameters to object
    key = a.name
    if key in selected_date_config:
        cfg = selected_date_config[key]
        a.selected_dates.setdefault("sar", {})
        a.selected_dates["sar"].update({
            "fft_per": cfg["fft_per"],
            "fft_lmin": cfg["fft_lmin"],
            "fft_lmax": cfg["fft_lmax"],
        })
        print(f"FFT tuning parameters saved to {a.name}")
    else:
        print(f"Warning: no FFT parameters config for AOI name: {a.name}")

# Prepare for JSON
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")
    