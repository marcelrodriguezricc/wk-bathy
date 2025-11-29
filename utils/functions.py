from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime, timedelta
import xarray as xr
import time, sys
import xml.etree.ElementTree as ET
import rasterio
import numpy as np


# Convert Lat/Lon Coordinate Format from Degrees, Minutes, Seconds to Decimal 
def dms_to_decimal(degrees, minutes, seconds, direction=None):

    # Mathematical conversion
    decimal = degrees + minutes / 60 + seconds / 3600

    # Make negative if direction is South or West
    if direction in ('S', 'W'):
        decimal *= -1
    
    # Round to six significant digits (10cm accuracy)
    decimal = round(decimal, 6)

    # Return decimal coordinate
    return decimal

# Download CRM based on link stored in AOI class
def download_crm(dods_url: str, out_dir: str = "data") -> Path:

    # Prepare string for download
    file_url = dods_url.replace("/dodsC/", "/fileServer/")

    # Set output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set filename
    local = out_dir / Path(file_url).name

    # Skip if filename exists
    if local.exists():
        print(f"Already exists: {local}")
        return local

    # Print download progress to console
    print(f"Downloading:\n  {file_url}\n→ {local}")
    start = time.time()
    def progress(blocks, block_size, total_size):
        elapsed = time.time() - start
        downloaded = blocks * block_size
        percent = downloaded / total_size * 100 if total_size > 0 else 0
        speed = downloaded / (1024 * elapsed) if elapsed > 0 else 0  # KB/s
        sys.stdout.write(
            f"\r[{percent:6.2f}%]  {downloaded/1e6:8.2f} MB "
            f"of {total_size/1e6:8.2f} MB | {speed:8.1f} KB/s | {elapsed:5.1f}s"
        )
        sys.stdout.flush()
    urlretrieve(file_url, local, reporthook=progress)
    print("\nDownload complete.")

    return local

# Based on input, iterate sequentially by n-amount in positive then negative directions
def iterate_offset(n):
    yield 0
    for k in range(1, n + 1):
        yield k
        yield -k

# Normalize Sentinel-1 href
def normalize_href(href: str) -> str:
    if href.startswith("s3://"):
        bucket_and_key = href[5:]
        bucket, key = bucket_and_key.split("/", 1)
        if bucket == "sentinel-s1-l1c":
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com/{key}"
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href

def _find_text_any_ns(root, local_name: str) -> str | None:
    """
    Find the text content of the first element with the given local name,
    ignoring XML namespaces.
    """
    el = root.find(f".//{{*}}{local_name}")
    return el.text if el is not None else None

def build_radar_coordinates(tif_path: str | Path, xml_path: str | Path):
    
    # Get path to both files
    tif_path = Path(tif_path)
    xml_path = Path(xml_path)

    # Get .tiff shape
    with rasterio.open(tif_path) as src:
        height = src.height # Number of azimuth lines
        width = src.width # Number of range samples

    # Prepare to parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get azimuth geometry
    az_start_str = _find_text_any_ns(root, "productFirstLineUtcTime")
    az_time_interval_text = _find_text_any_ns(root, "azimuthTimeInterval")
    if az_start_str is None or az_time_interval_text is None:
        raise RuntimeError("Could not find productFirstLineUtcTime or azimuthTimeInterval in XML.")
    az_time_interval = float(az_time_interval_text)  # seconds between lines
    az_start = datetime.fromisoformat(az_start_str.replace("Z", "+00:00"))

    # Get geolocation reference points
    geo_points = root.findall(".//{*}geolocationGridPoint")
    if not geo_points:
        raise RuntimeError("No geolocationGridPoint elements found in XML.")
    
    # Initialize arrays for storing pixel and slant times
    pixels = []
    slant_times = []

    # For each geolocation point...
    for gp in geo_points:
        pixel_text = gp.findtext(".//{*}pixel") # Get the associated pixel
        srt_text = gp.findtext(".//{*}slantRangeTime") # Get the associated slant range time
        if pixel_text is None or srt_text is None:
            continue
        pixels.append(int(pixel_text))
        slant_times.append(float(srt_text))

    # If there are is no available metadata, throw an error
    if len(pixels) == 0:
        raise RuntimeError("No pixel/slantRangeTime data found in geolocationGridPoint elements.")
    
    # Convert to numpy array to be operated on
    pixels = np.array(pixels)
    slant_times = np.array(slant_times)

    # Convert time to distance
    c = 299_792_458.0  # Speed of light
    slant_distances = 0.5 * c * slant_times # Slant range formula, converts to meters

    # Average slant range per unique pixel index
    unique_pix = np.unique(pixels)
    mean_slant = np.array([
        slant_distances[pixels == p].mean() for p in unique_pix
    ])

    # Interpolate to get a slant range value for every column
    cols = np.arange(width)
    slant_range_m = np.interp(cols, unique_pix, mean_slant)

    # Calculate azimuth time for every row
    rows = np.arange(height) 
    azimuth_time = np.array([
        az_start + timedelta(seconds=i * az_time_interval) for i in rows
    ])

    return slant_range_m, azimuth_time

# Get an interpolated latitude/longitude grid from georeference points
def build_lat_lon_grids(tif_path: str | Path, xml_path: str | Path):
    
    # Get path to both files
    tif_path = Path(tif_path)
    xml_path = Path(xml_path)

    # Get .tiff shape
    with rasterio.open(tif_path) as src:
        height = src.height # Number of azimuth lines
        width = src.width # Number of range samples

    # Prepare to parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get geolocation reference points
    geo_points = root.findall(".//{*}geolocationGridPoint")
    if not geo_points:
        raise RuntimeError("No geolocationGridPoint elements found in XML.")
    
    # Initialize arrays for storing pixel and slant times, latitudes and longitudes from each geolocation reference point
    lines = []
    pixels = []
    lats = []
    lons = []

    # For each geolocation reference point...
    for gp in geo_points:
        line_text = gp.findtext(".//{*}line") # Get row number associated with point in radar image
        pix_text = gp.findtext(".//{*}pixel") # Get associated pixel
        lat_text = gp.findtext(".//{*}latitude") # Get latitude
        lon_text = gp.findtext(".//{*}longitude") # Get longitude
        if None in (line_text, pix_text, lat_text, lon_text):
            continue
        lines.append(int(line_text))
        pixels.append(int(pix_text))
        lats.append(float(lat_text))
        lons.append(float(lon_text))
    if len(lines) == 0:
        raise RuntimeError("No valid line/pixel/lat/lon data in geolocationGridPoint elements.")

    # Convert to numpy array to be operated on
    lines = np.array(lines)
    pixels = np.array(pixels)
    lats = np.array(lats)
    lons = np.array(lons)

    # Build georeference point grid
    unique_lines = np.unique(lines)
    unique_pixels = np.unique(pixels)
    nL = len(unique_lines)
    nP = len(unique_pixels)
    lat_ref = np.full((nL, nP), np.nan, dtype=float)
    lon_ref = np.full((nL, nP), np.nan, dtype=float)
    line_index = {val: i for i, val in enumerate(unique_lines)}
    pixel_index = {val: j for j, val in enumerate(unique_pixels)}
    for line, pix, lat, lon in zip(lines, pixels, lats, lons):
        i = line_index[line]
        j = pixel_index[pix]
        lat_ref[i, j] = lat
        lon_ref[i, j] = lon

    # Interpolate latitude and longitude values across full image
    cols_full = np.arange(width)
    lat_on_tie_lines = np.empty((nL, width), dtype=float)
    lon_on_tie_lines = np.empty((nL, width), dtype=float)
    for i in range(nL):
        lat_on_tie_lines[i, :] = np.interp(cols_full, unique_pixels, lat_ref[i, :])
        lon_on_tie_lines[i, :] = np.interp(cols_full, unique_pixels, lon_ref[i, :])
    rows_full = np.arange(height)
    lat_full = np.empty((height, width), dtype=float)
    lon_full = np.empty((height, width), dtype=float)
    for j in range(width):
        lat_full[:, j] = np.interp(rows_full, unique_lines, lat_on_tie_lines[:, j])
        lon_full[:, j] = np.interp(rows_full, unique_lines, lon_on_tie_lines[:, j])

    return lat_full, lon_full

def find_data(type: str, data_dir: Path, name: str, filename: str) -> Path:
    pattern = f"{filename}_*_*.*"
    matches = sorted(
        m for m in data_dir.glob(pattern)
        if m.suffix.lower() in [".tif", ".tiff"] 
    )
    if not matches:
        raise FileNotFoundError(f"No {type} data found for AOI: {name}")
    if len(matches) > 1:
        print(f"Warning: multiple matches found, using: {matches[0].name}")
        for m in matches:
            print("  ", m)
    return matches[0]
