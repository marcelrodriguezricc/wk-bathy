# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from utils.functions import depth_from_lambda_T
from pathlib import Path
import pandas as pd
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform as rio_transform

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory and prefix for saving dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
opt_dir = ROOT_DIR / "data" / "st2" / "lambda"
opt_raster_dir = ROOT_DIR / "data" / "st2" / "subset"
sar_dir = ROOT_DIR / "data" / "st1" / "lambda"
sar_raster_dir = ROOT_DIR / "data" / "st1" / "subset"
depth_dir  = ROOT_DIR / "data" / "depth"
depth_dir.mkdir(parents=True, exist_ok=True)
sar_img_outdir = ROOT_DIR / "images" / "st1" / "depth"
sar_img_outdir.mkdir(parents=True, exist_ok=True)

min_depth = 0.0
max_depth = 150.0
max_abs_diff = 10

def filter_df(df):
    mask = (
        np.isfinite(df["h_disp"]) &
        np.isfinite(df["h_crm"])  &
        (df["h_disp"] > min_depth) & (df["h_disp"] < max_depth) &
        (df["h_crm"]  > 0.0) & (df["h_crm"]  < 200.0) & 
        (np.abs(df["h_disp"] - df["h_crm"]) < max_abs_diff)
    )
    return df.loc[mask].copy()

# For each AOI...
for a in aoi_list:
    opt_date = a.selected_dates["optical"]["date"]
    opt_path = opt_dir / f"{a.filename}_lambda_{opt_date}.csv"
    opt_raster_path = opt_raster_dir / f"{a.filename}_optical_subset_{opt_date}.tif"
    sar_date = a.selected_dates["sar"]["date"]
    sar_path = sar_dir / f"{a.filename}_lambda_{sar_date}.csv"
    sar_raster_path = sar_raster_dir / f"{a.filename}_sar_subset_{sar_date}.tiff"
    crm_path = ROOT_DIR / a.crm_local

    sar_df = pd.read_csv(sar_path)
    opt_df = pd.read_csv(opt_path)
    crm_path = Path(a.crm_local)
    ds_crm = xr.open_dataset(crm_path)

    tp_sar = a.selected_dates["sar"]["period"]
    tp_opt = a.selected_dates["optical"]["period"]

    depth_var = "z"
    lon_name = "lon" if "lon" in ds_crm.coords else "x"
    lat_name = "lat" if "lat" in ds_crm.coords else "y"

    sar_pts = xr.Dataset(
        {
            lon_name: (["points"], sar_df["lon"].values),
            lat_name: (["points"], sar_df["lat"].values),
        }
    )

    sar_h_crm = ds_crm[depth_var].interp(
        {lon_name: sar_pts[lon_name], lat_name: sar_pts[lat_name]},
        method="linear"
    )

    sar_df["h_crm"] = -1 * sar_h_crm.values
    
    with rasterio.open(opt_raster_path) as src_opt:
        opt_crs = src_opt.crs

        # Optical CSV currently has x,y in opt_crs
        xs = opt_df["x"].values
        ys = opt_df["y"].values

        # Transform from optical CRS to CRM CRS (likely EPSG:4326)
        lons, lats = rio_transform(opt_crs, "EPSG:4326", xs, ys)

        opt_df["lon"] = lons
        opt_df["lat"] = lats
    
    opt_pts = xr.Dataset(
    {
        lon_name: (["points"], opt_df["lon"].values),
        lat_name: (["points"], opt_df["lat"].values),
    }
    )

    opt_h_crm = ds_crm[depth_var].interp(
        {lon_name: opt_pts[lon_name], lat_name: opt_pts[lat_name]},
        method="linear"
    )

    opt_df["h_crm"] = -1 * opt_h_crm.values

    sar_df["h_disp"] = depth_from_lambda_T(sar_df["lambda_m"].values, T=tp_sar)

    # Optical depths from λ + Tp
    opt_df["h_disp"] = depth_from_lambda_T(opt_df["lambda_m"].values, T=tp_opt)
    
    sar_filt = filter_df(sar_df)
    opt_filt = filter_df(opt_df)


    print(f"{a.name}, SAR: {len(sar_df)} → {len(sar_filt)} filtered points")
    print(f"{a.name}, OPT: {len(opt_df)} → {len(opt_filt)} filtered points")
    
    with rasterio.open(sar_raster_path) as src_sar:
        sar_arr = src_sar.read(1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # SAR background
    vmin, vmax = np.percentile(sar_arr[0], (2, 98))
    ax.imshow(sar_arr, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(f"{a.name}, SAR-derived depths, ({sar_date})")

    # Scatter depth points (pixel coords i0 = x, j0 = y)
    sc = ax.scatter(
        sar_filt["i0"].values,           # x (cols)
        sar_filt["j0"].values,           # y (rows)
        c=sar_filt["h_disp"].values,     # color by depth
        s=20,
        cmap="viridis",
        edgecolors="none",
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth from SAR (m)")

    ax.set_axis_off()
    plt.tight_layout()

    # Save plot
    sar_img_outpath = sar_img_outdir / f"{a.filename}_sar_depth_{sar_date}.png"
    if sar_img_outpath.exists():
        print(f"Filtered depth map already exists for {a.name}, skipping save: {sar_img_outpath}")
    else:
        plt.savefig(sar_img_outpath, bbox_inches="tight", pad_inches=0)
        print(f"Saved filtered depth map for {a.name} → {sar_img_outpath}")

    
