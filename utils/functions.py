from pathlib import Path
from urllib.request import urlretrieve
import xarray as xr
import time, sys

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
        return href.replace(
        "s3://sentinel-s1-l1c/",
        "https://sentinel-s1-l1c.s3.us-west-2.amazonaws.com/")
    return href