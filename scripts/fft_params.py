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

# ----- FFT Parameters -----
# North Shore SAR
ekuhai_sar_per = 95
ekuhai_sar_lmin = 0.9
ekuhai_sar_lmax = 800

# North Shore Optical
ekuhai_opt_per = 90
ekuhai_opt_lmin = 90
ekuhai_opt_lmax = 300

# Golden Gate SAR
gg_sar_per = 95
gg_sar_lmin = 0.95
gg_sar_lmax = 800

# Golden Gate Optical
gg_opt_per = 90
gg_opt_lmin = 90
gg_opt_lmax = 300

# Diamond Shoals SAR
dshoals_sar_per = 85
dshoals_sar_lmin = 0.7
dshoals_sar_lmax = 400

# Diamond Shoals Optical
dshoals_opt_per = 90
dshoals_opt_lmin = 90
dshoals_opt_lmax = 300

# Puerto Rico SAR
pj_sar_per = 90
pj_sar_lmin = 0.7
pj_sar_lmax = 400

# Puerto Rico Optical
pj_opt_per = 70
pj_opt_lmin = 40
pj_opt_lmax = 300

# Compile params in dict for iterating
selected_date_config = {
    "North Shore, O'ahu, Hawaii": {
        "fft_sar_per": ekuhai_sar_per,
        "fft_sar_lmin": ekuhai_sar_lmin,
        "fft_sar_lmax": ekuhai_sar_lmax,
        "fft_opt_per": ekuhai_opt_per,
        "fft_opt_lmin": ekuhai_opt_lmin,
        "fft_opt_lmax": ekuhai_opt_lmax
    },
    "Golden Gate, California": {
        "fft_sar_per": gg_sar_per,
        "fft_sar_lmin": gg_sar_lmin,
        "fft_sar_lmax": gg_sar_lmax,
        "fft_opt_per": gg_opt_per,
        "fft_opt_lmin": gg_opt_lmin,
        "fft_opt_lmax": gg_opt_lmax
    },
    "Diamond Shoals, North Carolina": {
        "fft_sar_per": dshoals_sar_per,
        "fft_sar_lmin": dshoals_sar_lmin,
        "fft_sar_lmax": dshoals_sar_lmax,
        "fft_opt_per": dshoals_opt_per,
        "fft_opt_lmin": dshoals_opt_lmin,
        "fft_opt_lmax": dshoals_opt_lmax,
    },
    "Punta Jacinto, Puerto Rico": {
        "fft_sar_per": pj_sar_per,
        "fft_sar_lmin": pj_sar_lmin,
        "fft_sar_lmax": pj_sar_lmax,
        "fft_opt_per": pj_opt_per,
        "fft_opt_lmin": pj_opt_lmin,
        "fft_opt_lmax": pj_opt_lmax
    },
}

# For each AOI...
for a in aoi_list:
    
    # Save parameters to object
    key = a.name
    if key in selected_date_config:
        cfg = selected_date_config[key]
        a.selected_dates["sar"].update({
            "fft_per": cfg["fft_sar_per"],
            "fft_lmin": cfg["fft_sar_lmin"],
            "fft_lmax": cfg["fft_sar_lmax"],
        })
        a.selected_dates["optical"].update({
            "fft_per": cfg["fft_opt_per"],
            "fft_lmin": cfg["fft_opt_lmin"],
            "fft_lmax": cfg["fft_opt_lmax"],
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
    