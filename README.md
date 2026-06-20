# Computer Vision Ophthalmology Assistant

A classical computer vision pipeline for detecting tear film break-up regions in fluorescein-stained eye images, developed as a diagnostic support tool for clinicians screening patients for Dry Eye Syndrome (DES).

---

## Overview

Dry Eye Syndrome affects an estimated **16 million Americans** and is typically diagnosed through Break-Up Time (BUT) testing, where sodium fluorescein dye is applied to the eye and clinicians manually track the appearance of dark regions indicating tear film instability. This tool automates the visual analysis step, reducing reliance on subjective manual observation.

The pipeline operates in two stages:

1. **ROI Extraction**: Isolates the iris from raw eye images with sub-pixel consistency across varying capture angles
2. **Break-Up Region Detection**: Identifies and highlights candidate tear film break-up areas within the extracted ROI

Designed as a **clinician-facing support tool**, not a standalone classifier. Output is intended to direct expert attention, not replace it.

---

## Technical Pipeline

### Stage 1: Iris Segmentation

| Step | Technique | Purpose |
|------|-----------|---------|
| Histogram equalization | `cv2.equalizeHist()` | Normalize uneven pixel distributions |
| Gaussian blur | `cv2.GaussianBlur()` | Suppress high-frequency noise |
| Dark pixel suppression | Custom threshold (bottom 10%) | Prevent pupil from interfering with circle detection |
| Circle detection | Hough Circle Transform | Robustly detect iris boundary |
| Radius normalization | Mean radius/center across dataset | Standardize output across variable capture angles |

The Hough Circle Transform was selected after evaluating standard edge detectors (Canny, Sobel), which proved unreliable due to lighting variation and eyebrow interference. The mean-radius normalization step makes the pipeline dataset-agnostic: even if a new set of images is captured from a different angle or camera, the model adapts without retraining.

### Stage 2: Break-Up Region Detection

Operates on the cropped iris ROI:

- **Green channel extraction**: Fluorescein dye produces green-dominant signal; isolating this channel significantly improves signal-to-noise ratio
- **Histogram equalization + Gaussian blur**: Consistent normalization with Stage 1
- **Otsu's thresholding**: Automatic, image-adaptive threshold selection — no hardcoded intensity values
- **Morphological opening/closing**: Removes small noise artifacts that survive thresholding
- **Contour detection with size filtering**: Filters candidate regions by area, eliminating both noise (too small) and eyelash/edge artifacts (too large)

**Why contour over K-means**: K-means requires a pre-specified cluster count k, which forces artificial grouping when the true number of break-up regions varies per image. Contour detection is parameter-free in that respect and scales naturally to any number of regions.

---

## Results

- Iris detection: consistent and accurate across all test images, including samples captured at non-standard angles
- Break-up detection: successfully highlights candidate DES regions with tunable sensitivity
- False positives are expected and acknowledged; the tool is calibrated toward sensitivity over specificity, appropriate for a screening-assist context where missing a real region carries higher clinical cost than flagging a false one

---

## Design Philosophy

This system was deliberately built without machine learning to ensure:

- **Interpretability**: every decision in the pipeline is inspectable and explainable to a clinician
- **No training data dependency**: the pipeline generalizes to new datasets through parameter adjustment, not retraining
- **Speed**: inference is near-instantaneous on standard hardware

---

## Tech Stack

- Python
- OpenCV (`cv2`)
- NumPy

---

## Usage

```bash
git clone https://github.com/sjoeen/computer-vision-ophthalmology-assistant
cd computer-vision-ophthalmology-assistant
pip install opencv-python numpy
python main.py --input path/to/images/
```

Key tunable parameters:

| Parameter | Effect |
|-----------|--------|
| Radius shrink factor (default `0.69`) | Controls how much of the iris border is excluded to remove eyebrow noise |
| Contour area min/max | Controls sensitivity vs. specificity of break-up region detection |

---

## References

- National Eye Institute, *Dry Eye — At a Glance*, 2025. https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/dry-eye
- OpenCV Documentation, *Feature Detection & Image Processing*. https://docs.opencv.org/4.x/
- NumPy Documentation, *numpy.bitwise_and*. https://numpy.org/doc/2.2/
