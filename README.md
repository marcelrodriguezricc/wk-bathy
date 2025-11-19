# A quantitative evaluation of Optical vs. SAR-based Wave Kinematic Bathymetry (WKB) for deriving ocean depth in the surf-zone

## Project overview
The goal of this project is to derive bathymetry by WKB for four nearshore areas from both optical and SAR imagery, use existing hydrographic shallow water survey data as a source of ground-truth to compute the Root Mean Square Error (RMSE), and determine the strengths and weaknesses of each method.

- Step 1: Determine four Areas of Interest (AOI)

    - Select a variety of locations that feature a diverse set of conditions

    - For WKB, they must satisfy the conditions:
        - Publicly accessible hydrographic shallow water survey data
        - Swell-wave regime
            - Negligible effects from currents
        - An extended nearshore region of depths below 100 m

    -  And they should vary by...
        - Latitude (turbidity)
        - Exposure to marine processes (depositional/erosional)
        - Seafloor features (reefs, sandbars, canyons, heavy slope)

- Step 2: Load and normalize datasets

    - Initialize each AOI with central latitude and longitude, filename header, link to CRM, and bounding box extents.

    - Load CRM, extract important metadata and save in AOI object.

    - For a range of days around CRM creation date, use CMEMS Wave Analysis and Forecast to identify times for each AOI when Mean significant wave height greater than 1 m.
        - Average of the highest one-third (33%) of waves (measured from trough to crest) that occur in a given period.
        - Store swell period and direction data from CMEMS in AOI object for image selection and evaluation.
    
    - Look for Sentinel-2 imagery from days when significant wave height is greater than 1 m, and get image with best combination of factors for optical WKB.
        - Higher SWH, low cloud coverage, wave direction toward solar azimuth, preferable solar elevation. 
        - Store cloud coverage for image selection and evaluation.

    - Look for Sentinel-1 imagery from days when significant wave height is greater than 1 m, and get image with best combination of factors for SAR WKB.

    - Match the datum / CRS / projection of each image.

- Step 4: Derive bathymetry

    - Optical
        - Randon transform > discrete Fourier transform > wave celerity > linear dispersion [1]

    - SAR
        - 2D Fast Fourier transform > wavelength estimation > linear dispersion [2]

    - Generate map figures for each

- Step 5: Evaluation

    - Chart characteristics from wave model.

    - Compute root mean squared difference for each bathymetric derivation against CRM.

    - Generate a difference map using RMS error for each pixel.

    - Calculate & chart “visibility index” based on the unique requirements for identifying surface waves from optical and SAR imagery.
        - Sentinel 1 - Backscatter and L-min
        - Sentinel 2 - "Glint score" & cloud coverage
        
## Setup

Prereqs
- Python 3.11 (recommended)
- Git

Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Optional: install dev tooling
```bash
pip install pytest ruff black mypy
```

## Works cited

- [1] E. W. J. Bergsma, R. Almar, and P. Maisongrande, “Radon-Augmented Sentinel-2 Satellite Imagery to Derive Wave-Patterns and Regional Bathymetry,” Remote Sensing, vol. 11, no. 16,
p. 1918, Jan. 2019, doi: 10.3390/rs11161918.

- [2] S. D. Mudiyanselage, B. Wilkinson, and A. Abd-Elrahman,
“Automated High-Resolution Bathymetry from Sentinel-1 SAR Images in Deeper Nearshore Coastal Waters in Eastern Florida,” Remote Sensing, vol. 16, no. 1, p. 1, Jan. 2024, doi: 10.3390/rs16010001.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
© 2025 Marcel Rodriguez-Riccelli