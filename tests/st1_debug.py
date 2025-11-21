import rasterio
import xml.etree.ElementTree as ET



path = "/Users/marcel/Desktop/wkb-evaluation/data/st1/dshoals_sar_2023-08-26.tiff"
xml_path = "/Users/marcel/Desktop/wkb-evaluation/data/st1/dshoals_sar_2023-08-26.xml"

print("=== TIFF INFO ===")
with rasterio.open(path) as src:
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)
    print("Width, Height:", src.width, src.height)
    tiff_width, tiff_height = src.width, src.height

try:
    tree = ET.parse(xml_path)
    print("✅ XML parsed successfully.")
except Exception as e:
    print("❌ XML is missing or invalid:", e)

tree = ET.parse(xml_path)
root = tree.getroot()


found = root.find(".//productInformation")

if found is not None:
    print("Looks like Sentinel-1 annotation XML.")
else:
    print("XML parsed, but does not look like Sentinel-1 annotation.")

has_geo = root.find(".//geolocationGridPoint")
print("Has geolocation grid:", has_geo is not None)

with open(xml_path, "r") as f:
    for _ in range(10):
        print(f.readline().rstrip())