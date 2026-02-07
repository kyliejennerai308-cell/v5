# Vinyl Playmat Digital Restoration - v5

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

- **[CLEANUP_SCRIPT_PROGRESSION.md](CLEANUP_SCRIPT_PROGRESSION.md)** - Detailed comparison of all 4 versions
- **[REQUIREMENTS_VERIFICATION.md](REQUIREMENTS_VERIFICATION.md)** - Verification that v1 meets all requirements
- **[DEVELOPER_README.md](DEVELOPER_README.md)** - Technical documentation for developers
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start guide
- **[1- newclean-main/](1-%20newclean-main/)** - Latest production-ready script

---

## ⚙️ Requirements

- Python 3.8+
- opencv-python
- numpy

The `START_HERE.bat` script automatically installs dependencies on Windows.

**⚠️ RULE: No CLI flags permitted - launch via BAT file only**

---

## 🎨 Technical Details

**Latest version uses:**
- **8-color HLS palette** (more robust than HSV for lighting variations)
- **Advanced texture removal** (bilateral + guided filter + CLAHE + unsharp)
- **Edge preservation** (Canny edge detection with keep-out zones)
- **Text protection** (top-hat morphology + adaptive thresholding)
- **GPU acceleration** (automatic with fallback to CPU)
- **Zero configuration** (no CLI flags - run via START_HERE.bat only)

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