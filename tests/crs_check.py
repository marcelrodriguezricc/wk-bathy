import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pathlib import Path
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
crm_dir = ROOT_DIR / "data" / "crm"
sar_dir = ROOT_DIR / "data" / "st1" / "subset"
opt_dir = ROOT_DIR / "data" / "st2" / "selected"

crm_files = []
sar_files = []
opt_files = []

crm_files.extend(crm_dir.rglob("*.nc"))
sar_files.extend(sar_dir.rglob("*.tiff"))
opt_files.extend(opt_dir.rglob(".tif"))

info_list = []
print(sar_files)

for f in sar_files:
    try:
        with rasterio.open(f) as src:
            crs = src.crs.to_string() if src.crs else "None"
            res = src.res
            bounds = src.bounds
            width, height = src.width, src.height

            info_list.append({
                "file": f.name,
                "path": str(f),
                "crs": crs,
                "res": res,
                "bounds": bounds,
                "size": (width, height),
            })

            print(f"{f.name}")
            print(f"  CRS:        {crs}")
            print(f"  Resolution: {res}")
            print(f"  Size:       {width} x {height}")
            print(f"  Bounds:     {bounds}")
            print("")

    except Exception as e:
        print(f"ERROR reading {f}: {e}")
        info_list.append({"file": f.name, "error": str(e)})

for f in crm_files:
    try:
        with rasterio.open(f) as src:
            crs = src.crs.to_string() if src.crs else "None"
            res = src.res
            bounds = src.bounds
            width, height = src.width, src.height

            info_list.append({
                "file": f.name,
                "path": str(f),
                "crs": crs,
                "res": res,
                "bounds": bounds,
                "size": (width, height),
            })

            print(f"{f.name}")
            print(f"  CRS:        {crs}")
            print(f"  Resolution: {res}")
            print(f"  Size:       {width} x {height}")
            print(f"  Bounds:     {bounds}")
            print("")

    except Exception as e:
        print(f"ERROR reading {f}: {e}")
        info_list.append({"file": f.name, "error": str(e)})