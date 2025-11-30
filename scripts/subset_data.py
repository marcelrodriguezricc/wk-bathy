# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import rasterio 

import numpy as np
from utils.functions import build_radar_coordinates, build_lat_lon_grids, save_latlon
import matplotlib.pyplot as plt
from utils.data_classes import AOI
from pathlib import Path
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from dataclasses import asdict

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory and prefix for loading and saving datasets
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sar_dir = ROOT_DIR / "data" / "st1"
sar_outdir = ROOT_DIR / "data" / "st1" / "subset"
sar_outdir.mkdir(parents=True, exist_ok=True)
opt_dir = ROOT_DIR / "data" / "st2"
opt_outdir = ROOT_DIR / "data" / "st2" / "subset"
opt_outdir.mkdir(parents=True, exist_ok=True)
img_outdir = ROOT_DIR / "images" / "subset"
img_outdir.mkdir(parents=True, exist_ok=True)

# For each aoi...
for a in aoi_list:

    # Get paths for selected optical and sar images
    sar_date = a.selected_dates["sar"]["date"]
    sar_path = sar_dir / f"{a.filename}_sar_{sar_date}.tiff"
    xml_path = sar_dir / f"{a.filename}_sar_{sar_date}.xml"
    opt_date = a.selected_dates["optical"]["date"]
    opt_path = opt_dir / f"{a.filename}_optical_{opt_date}.tif"

    # Get bounding box min & max latitude & longitude for dataset query
    min_lon = float(a.lon) - float(a.bbox_lon)
    min_lat = float(a.lat) - float(a.bbox_lat)
    max_lon = float(a.lon) + float(a.bbox_lon)
    max_lat = float(a.lat) + float(a.bbox_lat)

    # Retrieve swell metadata
    swh_val = a.data[sar_date]["swh"]
    period_val = a.data[sar_date]["period"]
    direction_val = a.data[sar_date]["direction"]

    # --- SAR SUBSETTING ---

    # Retrieve SAR metadata
    pol = a.data[sar_date]["sar"]["polarization"]
    orb = a.data[sar_date]["sar"]["orbit_state"]

    # Use XML metadata to get slant range, azimuth time for each column/row, latitude, and longitude for each pixel of raw SAR measurement data
    slant_range_m, azimuth_time = build_radar_coordinates(sar_path, xml_path)
    lat_full, lon_full = build_lat_lon_grids(sar_path, xml_path)
    height, width = lat_full.shape

    # Mask for just values within AOI boundary box
    mask = (
        (lon_full >= min_lon) & (lon_full <= max_lon) &
        (lat_full >= min_lat) & (lat_full <= max_lat)
    )
    r_full = np.broadcast_to(slant_range_m, (height, width))
    r_aoi = r_full[mask]

    # Get mean cutoff wavelength for Velocity Brunching mechanism and store for evaluation
    mean_r = float(r_aoi.mean())
    velocity_st1 = 7100.0  # m/s
    mean_lmin = mean_r * np.sqrt(swh_val) / velocity_st1

    # Get subset of full image and update metadata
    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    lat_subset = lat_full[row_min:row_max + 1, col_min:col_max + 1]
    lon_subset = lon_full[row_min:row_max + 1, col_min:col_max + 1]

    with rasterio.open(sar_path) as src:
        window = Window.from_slices(
            (row_min, row_max + 1),
            (col_min, col_max + 1)
        )
        backscatter = src.read(1, window=window).astype("float32")
        new_meta = src.meta.copy()
        new_meta.update({
            "height": backscatter.shape[0],
            "width": backscatter.shape[1],
            "transform": src.window_transform(window),
            "dtype": "float32"
        })

    print(a.name, orb)
    # Flip horizontal and vertical if satellite is ascending, flip horizontal if descending, for correct image orientation
    if orb == "ascending":
        backscatter = backscatter[::-1, :]
        lat_subset = lat_subset[::-1, :]
        lon_subset = lon_subset[::-1, :]
    elif orb == "descending":
        backscatter = backscatter[:, ::-1]
        lon_subset = lon_subset[:, ::-1]

    save_latlon(lat_subset, lon_subset, a.name, a.filename, sar_date, sar_outdir)

    # Write subset to new raster
    sar_outpath = sar_outdir / f"{a.filename}_sar_subset_{sar_date}.tiff"
    if sar_outpath.exists():
        print(f"SAR subset already exists, skipping save: {sar_outpath}")  
    else:
        with rasterio.open(sar_outpath, "w", **new_meta) as dst:
            dst.write(backscatter, 1)
    
    # Save image of raster
    sar_img_outpath = img_outdir / f"{a.filename}_sar_subset_{sar_date}.png"
    if sar_img_outpath.exists():
        print(f"SAR image already exists for {a.name}, skipping save: {sar_img_outpath}")
    else:
        plt.imshow(backscatter, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.margins(0)
        plt.savefig(sar_img_outpath, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved optical image for {a.name} → {sar_img_outpath}")

    # --- OPTICAL SUBSETTING ---

    # Get subset of full image and update metadata
    with rasterio.open(opt_path) as src:
        bbox_native = transform_bounds(
            "EPSG:4326", src.crs,
            min_lon, min_lat, max_lon, max_lat
        )
        x_min, y_min, x_max, y_max = bbox_native
        window = from_bounds(
            x_min, y_min, x_max, y_max,
            transform=src.transform,
        )
        opt_data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            {
                "height": window.height,
                "width": window.width,
                "transform": src.window_transform(window),
            }
        )

    # Write subset to new raster
    opt_outpath = opt_outdir / f"{a.filename}_optical_subset_{opt_date}.tif"
    if opt_outpath.exists():
        print(f"Optical subset already exists for {a.name}, skipping save: {opt_outpath}")  
    else:
        with rasterio.open(opt_outpath, "w", **profile) as dst:
            dst.write(opt_data)

    # Save image of raster
    opt_img_outpath = img_outdir / f"{a.filename}_opt_subset_{opt_date}.png"
    if opt_img_outpath.exists():
        print(f"Optical image already exists for {a.name}, skipping save: {opt_img_outpath}")
    else:
        plt.imshow(opt_data[0], cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.margins(0)
        plt.savefig(opt_img_outpath, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved optical image for {a.name} → {opt_img_outpath}")

    # Store all relevant evaluation data with selected dates dict for each sensor type
    a.selected_dates["sar"] = {
        "date": sar_date,
        "swh": swh_val,
        "period": period_val,
        "direction": direction_val,
        "polarization": pol,
        "orbit_state": orb,
        "mean_slant_range": mean_r,
        "mean_lmin": mean_lmin,
    }
    cloud_val = a.data[opt_date]["optical"]["cloud_cover"]
    sun_az_val = a.data[opt_date]["optical"]["sun_azimuth"]
    sun_el_val = a.data[opt_date]["optical"]["sun_elevation"]
    a.selected_dates["optical"] = {
        "date": opt_date,
        "swh": swh_val,
        "period": period_val,
        "direction": direction_val,
        "cloud_cover": cloud_val,
        "sun_azimuth": sun_az_val,
        "sun_elevation": sun_el_val,
    }

# Prepare for JSON *after* modifying AOIs
payload = [asdict(a) for a in aoi_list]

# Save to JSON
out_path = Path("config/aoi_list.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Updated {len(payload)} AOIs → {out_path}")



