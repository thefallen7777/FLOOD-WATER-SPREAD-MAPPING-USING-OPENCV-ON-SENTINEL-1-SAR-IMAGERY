import os
import glob
import re
from difflib import SequenceMatcher

import cv2
import numpy as np
import rasterio

# =========================
# CONFIG (matches your folder structure)
# =========================
DATASET_ROOT = r"v1.2"
S1_DIR = os.path.join(DATASET_ROOT, "data", "flood_events", "HandLabeled", "S1Hand")
GT_DIR = os.path.join(DATASET_ROOT, "data", "flood_events", "HandLabeled", "LabelHand")

OUT_DIR = "outputs_flood_opencv"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def read_grayscale(path: str) -> np.ndarray:
    """
    Robust GeoTIFF reader for float/int rasters using rasterio (not OpenCV).
    Returns 2D float32 array.
    """
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)

    arr = arr.astype(np.float32)

    # multi-band -> mean, single band -> squeeze
    if arr.ndim == 3 and arr.shape[0] > 1:
        img = arr.mean(axis=0)
    else:
        img = arr.squeeze()

    # sanitize
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return img


def float_to_u8(img: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Convert float/int image to uint8 with optional percentile clipping.
    Works well for SAR-like ranges.
    """
    g = img.astype(np.float32)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # Percentile clip to avoid extreme outliers destroying contrast
    if clip_percentiles is not None:
        lo, hi = np.percentile(g, clip_percentiles)
        if hi > lo:
            g = np.clip(g, lo, hi)

    gmin, gmax = float(g.min()), float(g.max())
    if gmax > gmin:
        g = (g - gmin) * (255.0 / (gmax - gmin))
    else:
        g = np.zeros_like(g)

    return np.clip(g, 0, 255).astype(np.uint8)


def to_binary_mask(img_gray_float: np.ndarray) -> np.ndarray:
    """
    Otsu on SAR image:
    - Convert to uint8 safely
    - Otsu threshold
    - Decide inversion (flood usually darker => flood as white after inversion)
    Returns {0,255} uint8 mask.
    """
    img_u8 = float_to_u8(img_gray_float, clip_percentiles=(1, 99))
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cand1 = th
    cand2 = cv2.bitwise_not(th)

    def white_ratio(m):
        return float((m == 255).mean())

    r1, r2 = white_ratio(cand1), white_ratio(cand2)

    # flood usually not entire image; choose ratio closer to target
    target = 0.20
    mask = cand1 if abs(r1 - target) < abs(r2 - target) else cand2
    return mask


def postprocess(mask: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup: remove speckle and fill gaps.
    """
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k2, iterations=1)

    return cleaned


def ensure_same_size(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    return pred_mask


def binarize_gt(gt_gray_float: np.ndarray) -> np.ndarray:
    """
    Ground-truth binarization:
    Most label rasters are 0/background and >0 for flooded.
    Returns {0,255} uint8 mask.
    """
    g = np.nan_to_num(gt_gray_float.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # If it's already near-binary, this works perfectly.
    # If labels are something else, >0 still marks the labeled region.
    gt_bin = (g > 0).astype(np.uint8) * 255
    return gt_bin


def metrics(pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt are {0,255}. Treat 255 as flood=1, 0 as non-flood=0.
    Returns accuracy, precision, recall, f1, iou, tp, fp, fn, tn.
    """
    p = (pred == 255)
    g = (gt == 255)

    tp = int(np.logical_and(p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())

    eps = 1e-9
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)

    return float(acc), float(prec), float(rec), float(f1), float(iou), tp, fp, fn, tn


def overlay_mask_on_image(img_gray_float: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay red where mask=255 on a normalized grayscale image.
    Returns BGR uint8.
    """
    base = float_to_u8(img_gray_float, clip_percentiles=(1, 99))
    img_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    overlay = img_bgr.copy()
    overlay[mask == 255] = (0, 0, 255)  # red

    out = cv2.addWeighted(img_bgr, 0.75, overlay, 0.25, 0)
    return out


def normalize_key(path_or_name: str):
    """
    Normalize filename to improve S1<->GT matching.
    """
    base = os.path.splitext(os.path.basename(path_or_name))[0].lower()
    base = re.sub(r'[^a-z0-9]+', '_', base).strip('_')
    digits = "_".join(re.findall(r'\d+', base))
    return base, digits


def build_gt_index():
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.*")))
    idx = []
    for p in gt_files:
        b, d = normalize_key(p)
        idx.append((p, b, d))
    return idx


GT_INDEX = build_gt_index()


def best_match(s1_path: str):
    s1_base, s1_digits = normalize_key(s1_path)

    # 1) digits exact match (strong)
    digit_hits = [p for (p, b, d) in GT_INDEX if d and s1_digits and d == s1_digits]
    if len(digit_hits) == 1:
        return digit_hits[0]
    if len(digit_hits) > 1:
        scored = []
        for p in digit_hits:
            b, _ = normalize_key(p)
            scored.append((SequenceMatcher(None, s1_base, b).ratio(), p))
        scored.sort(reverse=True)
        return scored[0][1]

    # 2) fuzzy match on base
    scored = []
    for (p, b, d) in GT_INDEX:
        scored.append((SequenceMatcher(None, s1_base, b).ratio(), p))
    scored.sort(reverse=True)

    # accept threshold
    if scored and scored[0][0] >= 0.55:
        return scored[0][1]

    return None


# =========================
# MAIN LOOP
# =========================
s1_files = sorted(glob.glob(os.path.join(S1_DIR, "*.*")))
if not s1_files:
    raise RuntimeError(f"No images found in {S1_DIR}")

all_rows = []
summary = {"acc": [], "prec": [], "rec": [], "f1": [], "iou": []}

print("S1_DIR:", os.path.abspath(S1_DIR))
print("GT_DIR:", os.path.abspath(GT_DIR))
print("Found S1 files:", len(s1_files))
print("Found GT files:", len(GT_INDEX))
print("OUT_DIR:", os.path.abspath(OUT_DIR))
print("-" * 60)

for i, s1_path in enumerate(s1_files, start=1):
    gt_path = best_match(s1_path)
    if gt_path is None:
        print(f"[SKIP] No GT match for: {os.path.basename(s1_path)}")
        continue

    # Read rasters (float32)
    try:
        img = read_grayscale(s1_path)
    except Exception as e:
        print(f"[SKIP] Failed reading S1: {os.path.basename(s1_path)} -> {e}")
        continue

    try:
        gt_gray = read_grayscale(gt_path)
    except Exception as e:
        print(f"[SKIP] Failed reading GT: {os.path.basename(gt_path)} -> {e}")
        continue

    gt = binarize_gt(gt_gray)

    # Predict
    pred0 = to_binary_mask(img)
    pred = postprocess(pred0)
    pred = ensure_same_size(pred, gt)

    # Metrics
    acc, prec, rec, f1, iou, tp, fp, fn, tn = metrics(pred, gt)

    # Save visuals
    base = os.path.splitext(os.path.basename(s1_path))[0]
    overlay = overlay_mask_on_image(img, pred)

    cv2.imwrite(os.path.join(OUT_DIR, f"{base}_01_s1.png"), float_to_u8(img))
    cv2.imwrite(os.path.join(OUT_DIR, f"{base}_02_predmask.png"), pred)
    cv2.imwrite(os.path.join(OUT_DIR, f"{base}_03_gtmask.png"), gt)
    cv2.imwrite(os.path.join(OUT_DIR, f"{base}_04_overlay.png"), overlay)

    all_rows.append((base, acc, prec, rec, f1, iou, tp, fp, fn, tn))
    summary["acc"].append(acc)
    summary["prec"].append(prec)
    summary["rec"].append(rec)
    summary["f1"].append(f1)
    summary["iou"].append(iou)

    print(f"[{i}/{len(s1_files)}] {base}: IoU={iou:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

# =========================
# SUMMARY REPORT
# =========================
if not all_rows:
    raise RuntimeError("No pairs evaluated (could not match S1Hand files to LabelHand files by name).")

mean_acc  = float(np.mean(summary["acc"]))
mean_prec = float(np.mean(summary["prec"]))
mean_rec  = float(np.mean(summary["rec"]))
mean_f1   = float(np.mean(summary["f1"]))
mean_iou  = float(np.mean(summary["iou"]))

report_path = os.path.join(OUT_DIR, "metrics_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Flood Segmentation (OpenCV) - Metrics Summary\n")
    f.write("===========================================\n\n")
    f.write(f"Dataset root: {os.path.abspath(DATASET_ROOT)}\n")
    f.write(f"S1 folder   : {os.path.abspath(S1_DIR)}\n")
    f.write(f"GT folder   : {os.path.abspath(GT_DIR)}\n\n")

    f.write("Averages over matched image-mask pairs:\n")
    f.write(f"Accuracy : {mean_acc:.6f}\n")
    f.write(f"Precision: {mean_prec:.6f}\n")
    f.write(f"Recall   : {mean_rec:.6f}\n")
    f.write(f"F1-score : {mean_f1:.6f}\n")
    f.write(f"IoU      : {mean_iou:.6f}\n\n")

    f.write("Per-image results:\n")
    f.write("name\tacc\tprec\trec\tf1\tiou\tTP\tFP\tFN\tTN\n")
    for r in all_rows:
        base, acc, prec, rec, f1, iou, tp, fp, fn, tn = r
        f.write(f"{base}\t{acc:.6f}\t{prec:.6f}\t{rec:.6f}\t{f1:.6f}\t{iou:.6f}\t{tp}\t{fp}\t{fn}\t{tn}\n")

print("\nDONE.")
print(f"Saved outputs in: {os.path.abspath(OUT_DIR)}")
print(f"Metrics summary : {os.path.abspath(report_path)}")