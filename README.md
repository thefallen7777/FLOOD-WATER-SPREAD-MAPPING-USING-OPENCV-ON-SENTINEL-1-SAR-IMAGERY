# Flood / Water Spread Mapping using OpenCV and Sentinel-1 SAR

## Overview

This project implements a classical computer vision approach for flood detection using Sentinel-1 Synthetic Aperture Radar (SAR) imagery from the **Sen1Floods11 Essentials (Hand-Labeled) dataset**. The system segments flooded regions using intensity-based thresholding and morphological refinement, evaluates performance against manually annotated masks, and visualizes flood predictions for disaster management and construction risk assessment applications.

The objective is to demonstrate how lightweight OpenCV-based techniques can be applied for rapid flood monitoring using radar imagery.

---

## Dataset

Dataset used: **Sen1Floods11 Essentials – Hand-Labeled Subset (Kaggle)**

### Folder Structure
v1.2/
└── data/
└── flood_events/
└── HandLabeled/
├── S1Hand/ # Sentinel-1 SAR images (GeoTIFF)
└── LabelHand/ # Ground-truth flood masks


### Data Characteristics

#### S1Hand
- Sentinel-1 SAR backscatter intensity images
- Floating-point GeoTIFF format
- Single-channel radar data

#### LabelHand
- Manually annotated flood masks
- Pixel values:
  - 0 → Non-flood
  - >0 → Flood

---

## Understanding Image Identification

### 1️⃣ Original SAR Image (`*_01_s1.png`)

This is the normalized Sentinel-1 radar image.

- **Dark regions** → Low backscatter → Likely flood/water  
- **Bright regions** → High backscatter → Land, vegetation, buildings  

Water appears dark because smooth surfaces cause specular reflection of radar signals away from the sensor, resulting in low returned energy.

---

### 2️⃣ Predicted Flood Mask (`*_02_predmask.png`)

Binary segmentation result:

- **White pixels** → Predicted flood  
- **Black pixels** → Non-flood  

Generated using Otsu thresholding and morphological operations.

---

### 3️⃣ Ground Truth Mask (`*_03_gtmask.png`)

Manual annotations from dataset:

- **White** → Actual flood region  
- **Black** → Non-flood region  

Used for performance evaluation.

---

### 4️⃣ Overlay Image (`*_04_overlay.png`)

Final visualization combining original SAR and predicted mask:

- **Red regions** → Predicted flooded areas  
- Red on dark areas → Likely correct detection  
- Red on bright areas → Possible false positive  

This overlay enables quick visual validation of segmentation accuracy.

---

## Methodology

### 1. Preprocessing

- GeoTIFF images read using `rasterio`
- Float intensity values normalized to 8-bit range
- Percentile clipping reduces extreme values
- Gaussian smoothing reduces speckle noise

---

### 2. Flood Segmentation

- Otsu global thresholding applied
- Inversion logic ensures darker regions represent flood
- Morphological opening removes noise
- Morphological closing fills small gaps

Output: Binary flood mask (white = flood)

---

### 3. Evaluation Metrics

Predicted masks are compared against ground truth using:

- Accuracy
- Precision
- Recall
- F1-Score
- Intersection over Union (IoU)

**IoU Formula:**
IoU = TP / (TP + FP + FN)


These metrics quantify spatial overlap and classification performance.

---

## Output Structure

Generated results are saved in:
outputs_flood_opencv/


For each image:

- `_01_s1.png` → Original SAR image  
- `_02_predmask.png` → Predicted flood mask  
- `_03_gtmask.png` → Ground truth mask  
- `_04_overlay.png` → Flood highlighted in red  
- `metrics_summary.txt` → Performance report  

---

## Applications

- Flood-prone zone identification
- Disaster response support
- Infrastructure and construction risk analysis
- Rapid flood screening
- Preliminary geospatial hazard assessment

---

## Limitations

- Global thresholding may not adapt to complex terrain
- SAR speckle noise affects detection quality
- No advanced filtering (e.g., Lee filter) applied
- Not a deep learning model

This project demonstrates a classical computer vision baseline for flood mapping.

---

## Technologies Used

- Python 3.12
- OpenCV
- NumPy
- Rasterio
- Glob
- Difflib

---

## Future Improvements

- Apply speckle filtering (Lee/Frost filters)
- Implement adaptive thresholding
- Integrate deep learning models (e.g., U-Net)
- Add multi-temporal flood change detection
- Incorporate geospatial coordinate visualization
