# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from utils.functions import ll_dist, iter_windows
from pathlib import Path
import rasterio
import numpy as np
from scipy.ndimage import label, gaussian_filter, distance_transform_edt
import pandas as pd

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
data_dir = ROOT_DIR / "data" / "st1" / "masked"
sar_dir = ROOT_DIR / "data" / "st1" / "subset"
latlon_dir = ROOT_DIR / "data" / "st1" / "subset"
img_outdir = ROOT_DIR / "images" / "full_scene_fft"
img_outdir.mkdir(parents=True, exist_ok=True)
lambda_outdir = ROOT_DIR / "data" / "st1" / "lambda"
lambda_outdir.mkdir(parents=True, exist_ok=True)

# Local Parameters
window_size_m = 2000.0  
overlap = 0.8

# For each AOI...
for a in aoi_list:
    
    # Get date of SAR image and set path for retrieval based on date
    date = a.selected_dates["sar"]["date"]
    data_path = data_dir / f"{a.filename}_sar_masked_{date}.tiff"
    sar_path = sar_dir / f"{a.filename}_sar_subset_{date}.tiff"

    # Get period from object data
    tp = a.selected_dates["sar"]["period"]

    # Gravity coef.
    g = 9.81

    # ---- GET MEAN SPATIAL CHANGE IN METERS FOR LAT AND LON -----

    # Load interpolated latitude and longitude grid for SAR subset image pixels
    latlon_path = latlon_dir / f"{a.filename}_latlon_subset_{date}.npz"
    latlon = np.load(latlon_path)
    lat = latlon["lat"]
    lon = latlon["lon"]

    # Get dimensionality of latitude and longitude grid
    ny, nx = lat.shape

    # Use central row as reference
    j0 = ny // 2

    # Get latitude and longitude for all pixels except last in the row
    lat_row = lat[j0, :-1]
    lon_row = lon[j0, :-1]

    # Get latitude and longitude for all neighboring pixels
    lat_row_next = lat[j0, 1:]
    lon_row_next = lon[j0, 1:]

    # Find the X distance between neighboring pixels
    dxs = ll_dist(lat_row, lon_row, lat_row_next, lon_row_next)

    # Calculate mean of X distances
    dx = np.nanmean(dxs)

    # Use central column as reference
    i0 = nx // 2

    # Get latitude and longitude for all pixels except last in the column
    lat_col = lat[:-1, i0]
    lon_col = lon[:-1, i0]

    # Get latitude and longitude for all neighboring pixels 
    lat_col_next = lat[1:, i0]
    lon_col_next = lon[1:, i0]

    # Find Y distance between neighboring pixels
    dys = ll_dist(lat_col, lon_col, lat_col_next, lon_col_next)

    # Calculate mean of Y distances
    dy = np.nanmean(dys)

    # Determine number of pixels for window dimensions
    win_nx = int(window_size_m / abs(dx))
    win_ny = int(window_size_m / abs(dy))   

    # Make number odd so there is a clear center
    if win_nx % 2 == 0:
        win_nx += 1
    if win_ny % 2 == 0:
        win_ny += 1

    # Calculate the amount the center will shift from one window to the next
    step_x = max(1, int(win_nx * (1 - overlap)))
    step_y = max(1, int(win_ny * (1 - overlap)))

    # Open SAR image raster
    with rasterio.open(data_path) as src:

        # Assign data to variable
        sar_arr = src.read(1)

        window_centers = []
        wavelength_points = []

        # Iterate through each window
        for i, (window, cx, cy, j0, i0) in enumerate(
            iter_windows(sar_arr, lat, lon,
                win_nx, win_ny, step_x, step_y,
                max_nan_fraction = 0.3
            )
        ):

            # Get mask
            land_mask = np.isnan(window)
            ocean_mask = ~land_mask

            # Weighted feather based on distance from mask
            dist = distance_transform_edt(ocean_mask)
            r_feather = 40.0  # pixels
            weights = np.clip(dist / r_feather, 0.0, 1.0)
            mean_ocean = np.nanmean(window)
            window = np.where(land_mask, mean_ocean, window)

            # Use Hanning Window to taper the edges and reduce high frequency artifacts
            win_row = np.hanning(window.shape[0])[:, None]
            win_col = np.hanning(window.shape[1])[None, :]
            hann = win_row * win_col
            total_hann = weights * hann
            img_win = window * total_hann

            # Compute the 2D FFT, converting image to frequency domain
            F = np.fft.fft2(img_win)

            # Move DC bias to center, ocean swell radiates out from center, artifacts further from center
            F_shift = np.fft.fftshift(F)


            # Computer power spectrum to get wave energy from each bin and discard phase information
            P = np.abs(F_shift)**2
        
            # Get frequency of each FFT bin in the x and y direction, shift so centered at 0 (DC Offset), convert to wavenumber (radians/meter), and build as 2D grid
            kx = np.fft.fftshift(np.fft.fftfreq(win_nx, d=dx)) * 2 * np.pi
            ky = np.fft.fftshift(np.fft.fftfreq(win_ny, d=dy)) * 2 * np.pi
            kx2d, ky2d = np.meshgrid(kx, ky)

            # Calculate radial wavenumber (how rapidy ocean wave repeats in space)
            k = np.sqrt(kx2d**2 + ky2d**2)

            # Establish lower and upper threshold of wavelengths in consideration, and convert to wavenumber space
            lambda_deep = g * tp**2 / (2 * np.pi)
            lambda_min = 0.5 * lambda_deep
            lambda_max = 1.75 * lambda_deep
            k_min = 2 * np.pi / lambda_max
            k_max = 2 * np.pi / lambda_min

            # Apply threshold
            mask = (k > k_min) & (k < k_max)
    
            # Logarithmically compress 2D FFT power spectrum for thresholding and apply mask
            P_log = np.log10(P + 1e-12)
            P_smooth = gaussian_filter(P_log, sigma=1.0)
            P_band = P_smooth[mask]

            # Filter for only brightest 30% of pixels
            thr_percentile = a.selected_dates["sar"]["fft_per"]
            thr_value = np.percentile(P_band, thr_percentile)
            hot_mask = (P_log >= thr_value) & mask

            # Establish 3x3 grid structure for evaluating pixel connectedness
            structure = np.ones((3, 3), dtype=int)

            # Label groups of neighboring pixels that exceed threshold as connected 
            labels, n_labels = label(hot_mask, structure = structure)

            # Set threshold for blob size
            min_pixels = 5

            # Initialize array to store dict of blob information
            blob_info = []

            # Build pixel grid for calculating blob centroid in pixel space
            iy2d, ix2d = np.indices(P.shape)

            # For each labeled group...
            for lbl in range(1, n_labels + 1):
                region = (labels == lbl) # Mask for only pixels belonging to this blob
                npix = region.sum() # Get number of pixels in blob
                if npix < min_pixels: # If this blob has less pixels than our minimum blob size threshold...
                    continue # Skip it

                # Extract power and wavenumber values for pixels in this blob
                weights = P[region]
                kx_vals = kx2d[region]
                ky_vals = ky2d[region]
                k_vals  = k[region]

                # Get pixel space indices
                x_idx = ix2d[region]
                y_idx = iy2d[region] 

                # Calculate total power for blob
                total_power = weights.sum()

                # Calculate power-weighted centroid in k-space and pixel space
                W = total_power
                kx_c = np.sum(kx_vals * weights) / W
                ky_c = np.sum(ky_vals * weights) / W

                # Calculate magnitude of centroid k (rad/m)
                k_c = np.sqrt(kx_c**2 + ky_c**2)

                # Convert to wavelength (m)
                lambda_c = 2 * np.pi / k_c

                # Skip if the wavelength is unphysical (too long)
                if lambda_c > 1000:  # or > lambda_max, or whatever you choose
                    continue

                # Store info for blob as dict
                blob_info.append({
                    "label": int(lbl),
                    "npix": int(npix),
                    "total_power": float(total_power),
                    "kx_c": float(kx_c),
                    "ky_c": float(ky_c),
                    "k_c": float(k_c),
                    "lambda_m": float(lambda_c),
                })

            # Group blobs by sign of kx_c (left/right of origin) to enforce ±k pairing
            left_blobs  = [b for b in blob_info if b["kx_c"] < 0]
            right_blobs = [b for b in blob_info if b["kx_c"] > 0]

            if not left_blobs or not right_blobs:
                continue

            def blob_score(b, k_max, w=1.0):
                r_norm = b["k_c"] / k_max
                weight_center = 1.0 / (1.0 + w * r_norm)
                return b["total_power"] * weight_center

            # Pick highest ranked blob
            b1 = max(left_blobs,  key=lambda b: blob_score(b, k_max, w=1.0))

            wavelength = b1["lambda_m"]

            # Store pixel + geographic info and wavelength for this window
            wavelength_points.append({
                "j0": int(j0),
                "i0": int(i0),
                "lat": float(cx),
                "lon": float(cy),
                "lambda_m": float(wavelength),
                "total_power": float(b1["total_power"]),
            })

            if wavelength_points:
                df = pd.DataFrame(wavelength_points)
                csv_path = lambda_outdir / f"{a.filename}_lambda_{date}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved wavelengths for {a.name}, {date} → {csv_path}")
            else:
                print(f"No valid wavelength windows for {a.name}, {date}, nothing to export.")




            
