Flood / Water Spread Mapping using OpenCV and Sentinel-1 SAR
Overview

This project implements a classical computer vision approach for flood detection using Sentinel-1 Synthetic Aperture Radar (SAR) imagery from the Sen1Floods11 Essentials (Hand-Labeled) dataset. The system segments flooded regions using intensity-based thresholding and morphological refinement, evaluates performance against manually annotated masks, and visualizes flood predictions for disaster management applications.

The objective is to demonstrate how lightweight OpenCV-based techniques can be used for rapid flood assessment in construction and infrastructure planning scenarios.

Dataset

Dataset used: Sen1Floods11 Essentials – Hand-Labeled Subset (Kaggle)

Folder structure used in this project:

v1.2/
 └── data/
      └── flood_events/
           └── HandLabeled/
                ├── S1Hand/      # Sentinel-1 SAR images (GeoTIFF)
                └── LabelHand/   # Ground-truth flood masks
Data Characteristics

S1Hand

Sentinel-1 SAR backscatter intensity images

Stored as floating-point GeoTIFF

Single-channel radar data

LabelHand

Manually annotated binary flood masks

0 = Non-flood

0 = Flooded region

Why SAR for Flood Mapping?

Unlike optical imagery, SAR:

Works in all weather conditions

Is unaffected by cloud cover

Operates independently of daylight

In SAR imagery:

Darker regions → low backscatter → likely water/flood

Brighter regions → high backscatter → land, vegetation, buildings

Flooded surfaces appear dark due to specular reflection of radar signals away from the sensor.

Methodology
1. Preprocessing

GeoTIFF images are read using rasterio

Floating-point intensities are normalized to 8-bit format

Percentile clipping reduces extreme outliers

Image smoothing applied using Gaussian blur

2. Flood Segmentation

Otsu’s automatic thresholding is applied

Inversion logic ensures darker regions are classified as flood

Morphological operations:

Opening (noise removal)

Closing (gap filling)

Output:

Binary flood mask (white = flood, black = non-flood)

3. Evaluation

Predicted masks are compared with ground-truth annotations using:

Accuracy

Precision

Recall

F1-Score

Intersection over Union (IoU)

These metrics quantify segmentation performance.

4. Visualization

For each image, the system generates:

Normalized SAR image

Predicted flood mask

Ground truth mask

Overlay image (predicted flood highlighted in red)

Output Structure

Generated results are saved in:

outputs_flood_opencv/

For each sample:

_01_s1.png → Original SAR image

_02_predmask.png → Predicted flood mask

_03_gtmask.png → Ground truth mask

_04_overlay.png → Flood highlighted in red

metrics_summary.txt → Overall performance metrics

Applications

This system can support:

Disaster response planning

Flood-prone area identification

Infrastructure risk analysis

Construction project assessment

Rapid flood screening in emergency scenarios

Limitations

Uses global thresholding (Otsu), which may fail in heterogeneous terrain

SAR speckle noise affects segmentation quality

No advanced filtering (e.g., Lee filter) implemented

Performance varies across regions

This project serves as a baseline classical computer vision approach rather than a deep learning model.

Technologies Used

Python 3.12

OpenCV

NumPy

Rasterio

Glob

Difflib

Future Improvements

Apply speckle filtering (Lee/Frost filters)

Use adaptive thresholding

Integrate deep learning models (U-Net)

Add multi-temporal flood change detection module

Incorporate geospatial coordinate visualization
