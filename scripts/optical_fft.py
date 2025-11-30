# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
from utils.data_classes import AOI
from utils.functions import ll_dist
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label, gaussian_filter, distance_transform_edt


# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# Set directory for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
data_dir = ROOT_DIR / "data" / "st2" / "masked"
img_outdir = ROOT_DIR / "images" / "full_scene_fft"
img_outdir.mkdir(parents=True, exist_ok=True)

# For each AOI...
for a in aoi_list:

    # Get date of optical image and set path for retrieval based on date
    date = a.selected_dates["optical"]["date"]
    data_path = data_dir / f"{a.filename}_optical_masked_{date}.tif"

    # Load optical image raster
    with rasterio.open(data_path) as src:
        
        # ---- APPLY 2D FAST FOURIER TRANSFORM -----

        # Assign to variable as a Numpy array of float32 values
        img = src.read(1).astype(np.float32)

        # Get transform from metadata
        transform = src.transform

        # Assign 0 to NaN values from mask
        # img = np.where(np.isnan(img), 0, img)
        land_mask = np.isnan(img)
        ocean_mask = ~land_mask
        dist = distance_transform_edt(ocean_mask)
        r_feather = 40.0  # pixels
        weights = np.clip(dist / r_feather, 0.0, 1.0)  
        mean_ocean = np.nanmean(img)
        img = np.where(land_mask, mean_ocean, img)    

        # Remove the DC bias, or the constant brightness of optical image (average intensity) that is the zero-frequency component to reveal spectral content of waves
        img = img - mean_ocean

        # Use Hanning Window to taper the edges and reduce high frequency artifacts
        win_row = np.hanning(img.shape[0])[:, None]
        win_col = np.hanning(img.shape[1])[None, :]
        window = win_row * win_col
        total_window = weights * window
        img_win = img * total_window

        # Compute the 2D FFT, converting image to frequency domain
        F = np.fft.fft2(img_win)

        # Move DC bias to center, ocean swell radiates out from center, artifacts further from center
        F_shift = np.fft.fftshift(F)

        # Computer power spectrum to get wave energy from each bin and discard phase information
        P = np.abs(F_shift)**2

        # Get image size
        ny, nx = img_win.shape 

        # Apply log scale for visibility (power spectra span many orders of magnitude)
        P_log = np.log10(P + 1e-12)

        # Get pixel distance values from dataset metadata
        transform = src.transform
        crs = src.crs
        dx, dy = src.res
        dx = float(dx)
        dy = float(dy)

        # ----- CONVERT TO WAVENUMBER SPACE -----
        
        # Get frequency of each FFT bin in the x and y direction, shift so centered at 0 (DC Offset), convert to wavenumber (radians/meter), and build as 2D grid
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi
        kx2d, ky2d = np.meshgrid(kx, ky)

        # Calculate radial wavenumber (how rapidy ocean wave repeats in space)
        k = np.sqrt(kx2d**2 + ky2d**2)


        # ----- THRESHOLDING -----

        # Establish lower and upper threshold of wavelengths in consideration, and convert to wavenumber space
        lambda_min = 100
        lambda_max = a.selected_dates["optical"]["fft_lmax"]
        k_min = 2 * np.pi / lambda_max
        k_max = 2 * np.pi / lambda_min

        # Apply threshold
        mask = (k > k_min) & (k < k_max)
    
        # Logarithmically compress 2D FFT power spectrum for thresholding and apply mask
        P_log = np.log10(P + 1e-12)
        P_smooth = gaussian_filter(P_log, sigma=1.0)
        P_band = P_smooth[mask]

        # Filter for only brightest 30% of pixels
        thr_percentile = a.selected_dates["optical"]["fft_per"]
        thr_value = np.percentile(P_band, thr_percentile)
        hot_mask = (P_log >= thr_value) & mask

        # ----- DIVIDE INTO HIGH-INTENSITY CONTOUR BLOBS -----

        # Establish 3x3 grid structure for evaluating pixel connectedness
        structure = np.ones((3, 3), dtype=int)

        # Label groups of neighboring pixels that exceed threshold as connected 
        labels, n_labels = label(hot_mask, structure = structure)

        # Set threshold for blob size
        min_pixels = 10

        # Initialize array to store dict of blob information
        blob_info = []

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

            # Calculate total power for blob
            total_power = weights.sum()

            # Calculate power-weighted centroid in k-space
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

        # Sort blobs by total power from strongest to weakest
        blob_info_sorted = sorted(
            blob_info,
            key=lambda d: d["total_power"],
            reverse=True
        )
        
        # Calculate swell direction
        for b in blob_info_sorted:
            b["theta_rad"] = np.arctan2(b["ky_c"], b["kx_c"])
            b["theta_deg"] = np.degrees(b["theta_rad"])
            b["bearing_toward"] = (90.0 - b["theta_deg"]) % 360.0
            b["bearing_from"] = (b["bearing_toward"] + 180.0) % 360.0

        # Get wavelength and period
        g = 9.81
        for b in blob_info_sorted:
            lam = b["lambda_m"]
            b["period_s"] = np.sqrt((2 * np.pi / g) * lam)

    # Define zoom region
    zoom_frac = 0.30
    x_min = int((1 - zoom_frac) / 2 * nx)
    x_max = int((1 + zoom_frac) / 2 * nx)
    y_min = int((1 - zoom_frac) / 2 * ny)
    y_max = int((1 + zoom_frac) / 2 * ny)

    # Set up figure for plotting
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # 2D FFT
    axes[0].imshow(P_log, origin="lower", cmap="viridis")
    axes[0].set_title("2D FFT")
    axes[0].axis("off")

    # 2D FFT with Blobs
    axes[1].imshow(P_log, origin="lower", cmap="viridis")
    axes[1].set_title("2D FFT with blobs (zoomed)")
    axes[1].axis("off")
    
    # Set up legend for blobs plot
    legend_handles = [
        mpatches.Patch(color="red", label="K1 +"),
        mpatches.Patch(color="orange", label="K1 -"),
        mpatches.Patch(color="purple", label="K2 +"),
        mpatches.Patch(color="blue", label="K2 -")
    ]   
    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.8,
        borderpad=0.5
     )

    # Apply zoom
    axes[1].set_xlim([x_min, x_max])
    axes[1].set_ylim([y_min, y_max])

    # Show blob contours
    axes[1].contour(
        hot_mask.astype(int),
        levels=[0.5],
        colors="white",
        linewidths=0.7
    )

    # Plot primary two blobs with symbology
    top_four = blob_info_sorted[:4]
    colors = ["red", "orange", "purple", "blue"]
    for (b, col) in zip(top_four, colors):
        lbl = b["label"]
        axes[1].contour(
            (labels == lbl).astype(int),
            levels=[0.5],
            linewidths=1.2,
            colors=col,
        )

    # Get data from CMEMS to compare for tuning
    period_val = a.selected_dates["optical"]["period"]
    direction_val = a.selected_dates["optical"]["direction"]

    # Data from FFT for tuning
    period_kplus1 = round(blob_info_sorted[0]["period_s"], 1)
    direction_kplus1 = round(blob_info_sorted[0]["bearing_from"], 1)
    period_kminus1 = round(blob_info_sorted[1]["period_s"], 1)
    direction_kminus1 = round(blob_info_sorted[1]["bearing_from"], 1)
    period_kplus2 = round(blob_info_sorted[2]["period_s"], 1)
    direction_kplus2 = round(blob_info_sorted[2]["bearing_from"], 1)
    period_kminus2 = round(blob_info_sorted[3]["period_s"], 1)
    direction_kminus2 = round(blob_info_sorted[3]["bearing_from"], 1)

    # Comment box on side of plots for tuning
    fig.text(
            0.725, 0.5,                   
            f"CMEMS Period: {period_val:.2f} seconds\nCMEMS Mean Direction: {direction_val:.2f}°\nK1 + Period: {period_kplus1:.2f} seconds\nK1 + Direction: {direction_kplus1:.2f}\nK1 - Period: {period_kminus1:.2f} seconds\nK1 - Direction: {direction_kminus1:.2f}\nK2 + Period: {period_kplus2:.2f} seconds\nK2 + Direction: {direction_kplus2:.2f}\nK2 - Period: {period_kminus2:.2f} seconds\nK2 - Direction: {direction_kminus2:.2f}",
            va = "center", ha = "left",
            fontsize = 10,
            linespacing = 2.0,
            bbox=dict(facecolor = "white", edgecolor = "white")
        )
    
    # Move subplots to make space for comment box
    plt.subplots_adjust(left = 0.045, right = 0.70, top = 0.8, bottom = 0.05, wspace = 0.05)

    # Title over subplot titles
    fig.suptitle(
        f"{a.name} Full Scene FFT, Optical Imagery, {date}",
        fontsize = 14,
        fontweight = "bold",
        y = .87
    )

    # Save plot
    img_outpath = img_outdir / f"{a.filename}_opt_full_fft_{date}.png"
    if img_outpath.exists():
        print(f"Full FFT tuning plot already exists for {a.name}, skipping save: {img_outpath}")
    else:
        plt.savefig(img_outpath, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved FFT tuning plot for {a.name} → {img_outpath}")


