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
# North Shore SAR FULL
ns_sar_per = 95

# North Shore Optical FULL
ns_opt_per = 90

# North Shore SAR WINDOW
ns_wsar_per = 98

# North Shore Optical WINDOW
ns_wopt_per = 98

# Golden Gate SAR FULL
gg_sar_per = 95

# Golden Gate Optical FULL
gg_opt_per = 90

# Golden Gate SAR WINDOW
gg_wsar_per = 98

# Golden Gate Optical WINDOW
gg_wopt_per = 98

# Wassabo Beach SAR FULL
wb_sar_per = 85

# Wassabo Beach Optical FULL
wb_opt_per = 90

# Wassabo Beach SAR WINDOW
wb_wsar_per = 98

# Wassabo Beach Optical WINDOW
wb_wopt_per = 98

# Puerto Rico SAR FULL
pr_sar_per = 90

# Puerto Rico Optical FULL
pr_opt_per = 70

# Puerto Rico SAR WINDOW
pr_wsar_per = 98

# Puerto Rico Optical WINDOW
pr_wopt_per = 98

# Compile params in dict for iterating
selected_date_config = {
    "North Shore, O'ahu, Hawaii": {
        "fft_sar_per": ns_sar_per,
        "fft_opt_per": ns_opt_per,
        "fft_wsar_per": ns_wsar_per,
        "fft_wopt_per": ns_wopt_per,
    },
    "Golden Gate, California": {
        "fft_sar_per": gg_sar_per,
        "fft_opt_per": gg_opt_per,
        "fft_wsar_per": gg_wsar_per,
        "fft_wopt_per": gg_wopt_per,
    },
    "Wassabo Beach, Florida": {
        "fft_sar_per": wb_sar_per,
        "fft_opt_per": wb_opt_per,
        "fft_wsar_per": wb_wsar_per,
        "fft_wopt_per": wb_wopt_per,
    },
    "Rincon, Puerto Rico": {
        "fft_sar_per": pr_sar_per,
        "fft_opt_per": pr_opt_per,
        "fft_wsar_per": pr_wsar_per,
        "fft_wopt_per": pr_wopt_per,
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
            "fft_wper": cfg["fft_wsar_per"],
        })
        a.selected_dates["optical"].update({
            "fft_per": cfg["fft_opt_per"],
            "fft_wper": cfg["fft_wopt_per"],
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
    