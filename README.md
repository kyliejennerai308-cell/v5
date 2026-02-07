# Vinyl Playmat Digital Restoration

**Version:** 2.0 ("New Colour Regime")  
**Objective:** Automate the restoration of high-resolution vinyl scans, converting noisy, wrinkled material into clean, flat-color "vector-style" digital assets.

## 🚀 Quick Start

**Use the latest version:** `1- newclean-main/`

This directory contains the most refined and production-ready script for cleaning up scanned vinyl playmat images.

### ⚠️ IMPORTANT: BAT Launch Only - No CLI Flags
**This script does NOT accept command-line flags or arguments.**  
**You MUST run it via START_HERE.bat or directly without any arguments.**

### Usage (3 steps)
1. Copy your scanned images to: `1- newclean-main/scans/`
2. **Double-click:** `1- newclean-main/START_HERE.bat` (Windows) ✅ **RECOMMENDED**
   - OR run: `cd "1- newclean-main" && python restore_playmat_hsv.py` (no arguments)
3. Find cleaned images in: `1- newclean-main/scans/output/`

**❌ DO NOT run with flags:** `python restore_playmat_hsv.py --help` (this will error)  
**❌ DO NOT pass arguments:** `python restore_playmat_hsv.py scans/` (this will error)  
**✅ CORRECT:** Run via `START_HERE.bat` or `python restore_playmat_hsv.py` (no arguments)

---

## 📁 Repository Structure

This repository contains 4 versions of the cleanup script (1 = newest, 4 = oldest):

| Directory | Status | Description |
|-----------|--------|-------------|
| **1- newclean-main** | ✅ **USE THIS** | Latest version with new colors, advanced texture removal, zero grain |
| 2- repo2-main | 📦 Archive | HSV-based with parallel processing |
| 3- cleanup-main | 📦 Archive | HSV-based with upscaling |
| 4- old | 📦 Archive | Beta version (oldest) |
| Research | 📚 Reference | Sample images and research materials |

### 🔬 Analysis Tool

| Tool | Purpose |
|------|---------|
| **image_analysis.py** | Analyze scans to reveal edges, color shifts, and paint bleed |

See [IMAGE_ANALYSIS_README.md](IMAGE_ANALYSIS_README.md) for detailed usage instructions.

---

## 🎯 What It Does

The latest script (`1- newclean-main`) processes scanned playmat images to:

✅ **Silhouettes** = perfectly solid  
✅ **White logo fill** = clean  
✅ **Sky blue** = flat (no texture/grain)  
✅ **Text** = untouched, fully retained  
✅ **No grain** = completely removed  
✅ **Detail protection** = text/logos preserved  
✅ **Texture removal** = vinyl texture eliminated  
✅ **Color palette** = snapped to 8 exact colors  
✅ **Solid regions** = forced throughout  
✅ **Edges** = reinserted and cleaned  
✅ **Straight lines** = cleaned where present  
✅ **Filled colored** = every pixel assigned  

---

## 📖 Documentation

- **[V2_UPDATE_SUMMARY.md](V2_UPDATE_SUMMARY.md)** - v2.0 documentation update summary and migration path
- **[CLEANUP_SCRIPT_PROGRESSION.md](CLEANUP_SCRIPT_PROGRESSION.md)** - Detailed comparison of all 4 versions
- **[REQUIREMENTS_VERIFICATION.md](REQUIREMENTS_VERIFICATION.md)** - Verification that v1 meets all requirements
- **[DEVELOPER_README.md](DEVELOPER_README.md)** - Technical documentation for developers
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start guide
- **[IMAGE_ANALYSIS_README.md](IMAGE_ANALYSIS_README.md)** - Image analysis tool documentation
- **[1- newclean-main/](1-%20newclean-main/)** - Latest production-ready script

---

## ⚙️ Requirements

- Python 3.8+
- opencv-python
- numpy

**Installation:**
```bash
pip install -r requirements.txt
```

The `START_HERE.bat` script automatically installs dependencies on Windows.

**⚠️ RULE: No CLI flags permitted - launch via BAT file only**

---

## 🎨 Technical Details

### Current Implementation (v1)
**Latest version uses:**
- **8-color HLS palette** (more robust than HSV for lighting variations)
- **Advanced texture removal** (bilateral + guided filter + CLAHE + unsharp)
- **Enhanced edge preservation** (Canny edge detection + color gradient analysis with keep-out zones)
- **LAB color space analysis** (A/B channel analysis for paint bleed detection)
- **Color distance mapping** (Sobel gradient magnitude to identify color shift boundaries)
- **Text protection** (top-hat morphology + adaptive thresholding + LAB-based white detection)
- **K-means posterization** (reduce to exact palette colors for cleaner masks)
- **Morphological dust removal** (opening operation to eliminate scan artifacts)
- **Watershed segmentation** (split touching shapes of same color)
- **Mask-based repainting** (edge-respecting solid color fills)
- **GPU acceleration** (automatic with fallback to CPU)
- **Zero configuration** (no CLI flags - run via START_HERE.bat only)

### Target Specification (v2.0)
The project aims to achieve these technical goals:

**Environment:**
- Python 3 with OpenCV (cv2), NumPy, CuPy (optional)
- CUDA hardware acceleration support via OpenCV CUDA modules or CuPy
- BGR color space for OpenCV compatibility
- Input: High-res JPG scans → Output: Lossless PNG

**10-Color Master Palette:**
| Color | BGR Value | Role/Usage |
|-------|-----------|------------|
| Sky Blue | `[233, 180, 130]` | Background canvas (perfectly flat) |
| Hot Pink | `[205, 0, 253]` | "STEPS" logo, footprints, number backings |
| Bright Yellow | `[1, 252, 253]` | Main fill for silhouettes and ladder rungs |
| Pure White | `[255, 255, 255]` | Logo interiors, stars (protected details) |
| Neon Green | `[0, 213, 197]` | Thin border around yellow figures |
| Dark Purple | `[140, 0, 180]` | Outermost thin stroke on logos |
| Vibrant Red | `[1, 13, 245]` | Thin underlines and ladder accents |
| Deep Teal | `[10, 176, 149]` | Instructional text and secondary shadows |
| Secondary Yellow | `[55, 255, 255]` | Secondary fill for group silhouettes |
| Black | `[0, 0, 0]` | Dead space or scan document edges |

**Processing Pipeline:**
1. **Phase 1: Pre-Processing & "Protected Mode"**
   - Mask high-contrast features (text, stars, logos) to prevent erosion
   - Suppress shadows/glare (brighter or darker than Sky Blue)

2. **Phase 2: Background & Silhouette Restoration**
   - Flood-fill background with solid Sky Blue
   - Area-based filtering (remove noise < 20 pixels using `connectedComponentsWithStats`)
   - Restore original green outline with protected cleanup

3. **Phase 3: Final Assembly (Layering Order)**
   - Bottom: Flat Sky Blue Background
   - Middle-Lower: Cleaned Original Green Outline
   - Middle-Upper: Cleaned Yellow Silhouette
   - Top: White interiors and Pink/Purple logo strokes

**Developer Constraints:**
- ⚠️ Use **3×3 kernels only** (not 5×5) to prevent melting small text or rounding star points
- ⚠️ Keep `approxPolyDP` epsilon at **0.001** (not 0.02) for organic silhouette shapes
- ⚠️ Apply anti-aliasing only to color boundaries to prevent stair-step pixelation
- ⚠️ Avoid heavy morphological opening on instructional text to maintain legibility

See [CLEANUP_SCRIPT_PROGRESSION.md](CLEANUP_SCRIPT_PROGRESSION.md) for full technical comparison of all versions.

---

## 🔧 Troubleshooting

**Issue:** Window closes immediately  
**Solution:** Make sure you have images in `1- newclean-main/scans/` folder first

**Issue:** Python not found  
**Solution:** Install Python 3.8+ from python.org and check "Add Python to PATH"

**Issue:** Colors look wrong  
**Solution:** Use `1- newclean-main` (latest version) - older versions have inferior color handling

---

**Ready?** Copy your images to `1- newclean-main/scans/` and run `START_HERE.bat`!