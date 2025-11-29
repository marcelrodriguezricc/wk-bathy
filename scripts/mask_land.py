# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import numpy as np
import shapely as shp
from shapely.geometry import box
from shapely import affinity
import matplotlib.pyplot as plt

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory and prefix for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
land_dir = ROOT_DIR / "data" / "land_vectors"
land_path =  land_dir / "ne_10m_shp"
sar_dir = ROOT_DIR / "data" / "st1" / "subset"
sar_outdir = ROOT_DIR / "data" / "st1" / "masked"
sar_outdir.mkdir(parents=True, exist_ok=True)
xml_dir = ROOT_DIR / "data" / "st1"
opt_dir = ROOT_DIR / "data" / "st2" / "subset"
opt_outdir = ROOT_DIR / "data" / "st2" / "masked"
opt_outdir.mkdir(parents=True, exist_ok=True)
img_outdir = ROOT_DIR / "images" / "masked"
img_outdir.mkdir(parents=True, exist_ok=True)


# For each AOI...
for a in aoi_list:

    # Get bounding box min & max latitude & longitude for dataset query
    min_lon = float(a.lon) - float(a.bbox_lon)
    min_lat = float(a.lat) - float(a.bbox_lat)
    max_lon = float(a.lon) + float(a.bbox_lon)
    max_lat = float(a.lat) + float(a.bbox_lat)

    # Initialize data array to store optical and SAR
    data = []

    # Get dates for each sensor dataset
    sar_date = a.selected_dates["sar"]["date"]
    opt_date = a.selected_dates["optical"]["date"]

    # Reconstruct filepaths for SAR image, Numpy Array, XML metadata, and optical imagery
    sar_path = sar_dir / f"{a.filename}_sar_subset_{sar_date}.tiff"
    npz_path = sar_dir / f"{a.filename}_latlon_subset_{sar_date}.npz"
    xml_path = xml_dir / f"{a.filename}_sar_{sar_date}.xml"
    opt_path = opt_dir / f"{a.filename}_optical_subset_{opt_date}.tif"

    # Add both SAR and optical paths to data array for loop
    data.append(sar_path)
    data.append(opt_path)

    # Open land vector
    land = gpd.read_file(land_path)

    # Convert to EPSG 4326 and apply bounding box
    land = land.to_crs("EPSG:4326")
    aoi_poly = box(min_lon, min_lat, max_lon, max_lat)
    aoi_poly_big = affinity.scale(aoi_poly, xfact=2, yfact=2, origin='center')
    aoi_df = gpd.GeoDataFrame(geometry=[aoi_poly_big], crs="EPSG:4326")
    clipped_land = gpd.clip(land, aoi_df)

    # Convert to EPSG 32610 to add buffer, then convert back
    land_utm = clipped_land.to_crs("EPSG:32610")
    land_buffered = land_utm.buffer(a.shoreline_buffer)
    land_buffered = land_buffered.to_crs("EPSG:4326")

    # Loop for both dataset (optical and SAR)...
    for ds in data:

        # Open dataset
        with rasterio.open(ds) as src:

            # IF the image has no CRS (SAR)...
            if src.crs == None:
                sar_data = src.read(1).astype("float32")

                # Copy metadata to new dataset
                sar_profile = src.meta.copy()

                # Load interpolated latitude & longitude grid for SAR image pixels from file and assign to variables
                latlon = np.load(npz_path)
                lat_subset = latlon["lat"]
                lon_subset = latlon["lon"]

                # Create two 1D arrays to iterate through based on lat_subset shape
                rows, cols = lat_subset.shape

                # Create a 2D pixel array that will serve as the land mask
                land_mask = np.zeros((rows, cols), dtype=bool)

                # Union all land geometry so coordinates can be tested against the closed polygon
                land_geom = land_buffered.union_all()

                # Create mask for coincident latitude and longitude points from polygon and SAR image pixels
                land_mask = shp.contains_xy(land_geom, lon_subset, lat_subset)
    
                # Apply mask to SAR data
                sar_masked = np.where(land_mask, np.nan, sar_data)

                # Update dataset profile
                sar_profile.update(
                    dtype = "float32",
                    nodata = np.nan,
                    height = sar_masked.shape[0],
                    width = sar_masked.shape[1],
                )
                
                # Save image with percent stretch, unmasked area in red, for debugging and establishing buffer extent
                sar_masktest_outpath = img_outdir / f"{a.filename}_sar_masktest_{sar_date}.png"
                if sar_masktest_outpath.exists():
                    print(f"SAR mask test image already exists for {a.name}, skipping save: {sar_masktest_outpath}")
                else:
                    vmin, vmax = np.percentile(sar_data, (2, 98))
                    plt.imshow(sar_data, cmap = "gray", vmax = vmax, vmin = vmin)
                    plt.imshow(sar_masked, cmap = "Reds", alpha=0.3, vmax = vmax, vmin = vmin)
                    plt.axis("off")
                    plt.tight_layout(pad = 0)
                    plt.margins(0)
                    plt.savefig(sar_masktest_outpath, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"Saved SAR mask test image for {a.name} → {sar_masktest_outpath}")

                # Save masked image with no modifications
                sar_mask_outpath = img_outdir / f"{a.filename}_sar_mask_{sar_date}.png"
                if sar_mask_outpath.exists():
                    print(f"Masked SAR image already exists for {a.name}, skipping save: {sar_mask_outpath}")
                else:
                    plt.imshow(sar_masked, cmap = "gray")
                    plt.axis("off")
                    plt.tight_layout(pad = 0)
                    plt.margins(0)
                    plt.savefig(sar_mask_outpath, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"Saved masked SAR image for {a.name} → {sar_mask_outpath}")
                
                # Write subset to new raster
                sar_data_outpath = sar_outdir / f"{a.filename}_sar_subset_{sar_date}.tiff"
                if sar_data_outpath.exists():
                    print(f"Masked SAR data already exists, skipping save: {sar_data_outpath}")
                else:
                    with rasterio.open(sar_data_outpath, "w", **sar_profile) as dst:
                        dst.write(sar_masked, 1)
                
            # Else if the image has a  CRS (optical)...
            else: 

                # Load the optical data for visualizing the mask, to ensure the buffer has fully included all land in imagery
                opt_data = src.read().astype("float32")
                opt_profile = src.profile.copy()

                # Reproject the land polygon to the same CRS as imagery (UTM based on location)
                land_reproj = land_buffered.to_crs(src.crs)

                # Union all land geometry so the mask can be applied
                land_geom = land_reproj.union_all

                # Mask the optical imagery with the land vector
                opt_masked, opt_transform = mask(
                    src,
                    land_reproj.geometry,
                    invert = True,
                    nodata = src.nodata
                )

                # Set optical mask values to NaN
                opt_masked = opt_masked.astype('float32')
                opt_masked[opt_masked == src.nodata] = np.nan

                # Update dataset profile
                opt_profile.update(
                    dtype = "float32",
                    nodata = np.nan,
                    height = opt_masked.shape[1],
                    width = opt_masked.shape[2],
                    transform = opt_transform,
                )

                # Save image with percent stretch, unmasked area in red, for debugging and establishing buffer extent
                opt_masktest_outpath = img_outdir / f"{a.filename}_opt_masktest_{opt_date}.png"
                if opt_masktest_outpath.exists():
                    print(f"Optical mask test image already exists for {a.name}, skipping save: {opt_masktest_outpath}")
                else:
                    vmin, vmax = np.percentile(opt_data[0], (2, 98))
                    plt.imshow(opt_data[0], cmap = "gray", vmax = vmax, vmin = vmin)
                    plt.imshow(opt_masked[0], cmap = "Reds", alpha = 0.3)
                    plt.axis("off")
                    plt.tight_layout(pad = 0)
                    plt.margins(0)
                    plt.savefig(opt_masktest_outpath, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"Saved optical mask test image for {a.name} → {opt_masktest_outpath}")
                
                # Save masked image with no modifications
                opt_mask_outpath = img_outdir / f"{a.filename}_opt_mask_{opt_date}.png"
                if opt_mask_outpath.exists():
                    print(f"Optical mask test image already exists for {a.name}, skipping save: {opt_mask_outpath}")
                else:
                    plt.imshow(opt_masked[0], cmap = "gray")
                    plt.axis("off")
                    plt.tight_layout(pad = 0)
                    plt.margins(0)
                    plt.savefig(opt_mask_outpath, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"Saved masked optical image for {a.name} → {opt_mask_outpath}")
                
                # Write subset to new raster
                opt_data_outpath = opt_outdir / f"{a.filename}_optical_masked_{opt_date}.tif"
                if opt_data_outpath.exists():
                    print(f"Masked optical data already exists for {a.name}, skipping save: {opt_data_outpath}")  
                else:
                    with rasterio.open(opt_data_outpath, "w", **opt_profile) as dst:
                        dst.write(opt_masked)