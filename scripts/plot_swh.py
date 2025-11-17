# Compile a list of dates and times where significant wave height greater than 1M

# Run from root
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Libraries
import json
import xarray as xr
import matplotlib.pyplot as plt
from utils.data_classes import AOI, params
from utils.functions import iterate_offset
from datetime import datetime, timedelta
from pathlib import Path

# Load JSON array
with open("config/aoi_list.json") as f:
    aoi_data = json.load(f)

# Reconstruct AOI objects and store in array
aoi_list = [AOI(**a) for a in aoi_data]

# User Parameters
num_days = params.num_days

# Set directory for loading dataset
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
dir = ROOT_DIR / "data" / "models"

# Set output folder for plots
outdir_img = ROOT_DIR / "images" / "swh_plots"
outdir_img.mkdir(parents=True, exist_ok=True)

# Initialize ict to store date and SWH mean arrays
results = {} 

# For each AOI...
for a in aoi_list:
    # Fetch center date
    center_date = datetime.fromisoformat(str(a.crm_date))

    # Get filename for loading dataset
    header = Path(a.filename).stem

    # Initialize arrays to store dates and SWH means greater than 1m
    dates = []
    swh_means = []

    # For each day "num_days" before and after the center date...
    for offset in iterate_offset(num_days):

        # Get day based on current offset
        d = (center_date + timedelta(days=offset) + timedelta(days=a.date_offset)).date()

        # Get path to wave model for day and load
        path = dir / f"{header}_wave_{d}.nc"

        # Load dataset, get significant wave height, and calculate daily mean
        ds = xr.open_dataset(path)
        swh = ds["VHM0"]
        swh_avg = float(swh.mean().values)

        # Convert datetime to string
        date_str = d.strftime("%Y-%m-%d")

        # Store date and SWH in respective arrays
        dates.append(date_str)
        swh_means.append(swh_avg)

    # Establish plot size, parameters
    fig = plt.figure(figsize=(10, 4))
    plt.plot(dates, swh_means, marker="o", label="Mean SWH")

    # Symbology for dates with SWH greater than 1
    dates_gt1 = [d for d, v in zip(dates, swh_means) if v > 1]
    vals_gt1  = [v for v in swh_means if v > 1]
    plt.scatter(
        dates_gt1, vals_gt1,
        s=60,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="> 1m"
    )

    # Set filename from AOI object for saving plot and append to path string
    fname_stem = Path(a.filename).stem
    outpath = outdir_img / f"{fname_stem}_swh.png"

    # Initialize plot and format
    plt.xlabel("Date")
    plt.ylabel("Mean SWH")
    plt.title(f"Daily Mean SWH – {a.name}")
    plt.legend(loc="upper left")
    plt.gcf().autofmt_xdate(rotation=45)
    plt.xticks(fontsize=8)
    plt.tight_layout()

    # Save the figure, unless already in folder
    if outpath.exists():
            print(f"Image already exists, skipping save: {outpath}")  
    else:  
        fig.savefig(outpath, dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()