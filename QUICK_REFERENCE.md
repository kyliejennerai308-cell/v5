# Quick Reference: Using the Latest Cleanup Script

## ⚡ Ultra-Fast Guide

⚠️ **IMPORTANT:** No CLI flags permitted - run via BAT file only

```bash
# 1. Navigate to latest version
cd "1- newclean-main"

# 2. Put your scans in scans/ folder

# 3. Run via BAT file (REQUIRED)
START_HERE.bat     # Windows - DOUBLE-CLICK THIS

# OR run directly without arguments
python restore_playmat_hsv.py  # No flags, no arguments

# 4. Get output from scans/output/
```

**❌ FORBIDDEN:**
```bash
python restore_playmat_hsv.py --help          # ERROR
python restore_playmat_hsv.py --workers 4     # ERROR  
python restore_playmat_hsv.py scans/          # ERROR
python restore_playmat_hsv.py --use-gpu       # ERROR
```

**✅ CORRECT:**
```bash
START_HERE.bat                                # ✓ Windows
python restore_playmat_hsv.py                 # ✓ Linux/Mac (no args)
```

---

## 🎯 What You Get

**Input:** Scanned JPEG with wrinkles, glare, vinyl texture, grain  
**Output:** Perfect PNG with 8 exact colors, zero grain, flat regions

---

## ✨ Features of v1 (Latest)

| Feature | Status | Implementation |
|---------|--------|----------------|
| **No grain** | ✅ | Bilateral + guided filter |
| **Flat sky blue** | ✅ | Multi-stage texture removal |
| **Clean white logos** | ✅ | Top-hat + adaptive threshold |
| **Protected text** | ✅ | Edge detection + keep-out zones |
| **Solid silhouettes** | ✅ | Morphological close + area-open |
| **Clean straight lines** | ✅ | Unsharp + Canny edges |
| **8 exact colors** | ✅ | HLS palette snapping |
| **No bleeding** | ✅ | Conditional dilation |

---

## 🎨 8-Color Master Palette

The latest script outputs only these 8 exact colors:

1. **BG_SKY_BLUE** - Background (flat canvas)
2. **PRIMARY_YELLOW** - Silhouettes/bright features
3. **HOT_PINK** - Primary logo/footprints
4. **DARK_PURPLE** - Outer borders (3rd layer)
5. **PURE_WHITE** - Stars/logo interior (protected)
6. **STEP_RED_OUTLINE** - Ladder accents/underlines
7. **LIME_ACCENT** - Silhouette outlines
8. **DEAD_BLACK** - Void/scan edges

**Every pixel** is assigned to one of these 8 colors. No gradients, no noise, no in-between values.

---

## 📊 Processing Pipeline

```
Scanned JPEG
    ↓
[1] Bilateral Filter ───────→ Remove vinyl texture
    ↓
[2] Guided Filter ──────────→ Edge-aware smoothing
    ↓
[3] Auto-Gamma ─────────────→ Normalize exposure
    ↓
[4] CLAHE ──────────────────→ Enhance local contrast
    ↓
[5] Unsharp Mask ───────────→ Sharpen edges
    ↓
[6] HLS Color Conversion ───→ Better color detection
    ↓
[7] Edge Detection ─────────→ Canny keep-out zones
    ↓
[8] White Text Detection ───→ Top-hat + adaptive
    ↓
[9] Dark Outline Detection ─→ Invert-L trick
    ↓
[10] Color Mask Creation ───→ 8 separate masks
    ↓
[11] Morphological Ops ─────→ Close + area-open
    ↓
[12] Priority Assignment ───→ Resolve overlaps
    ↓
[13] Nearest-Color Fallback → Fill remaining pixels
    ↓
Perfect PNG (8 colors, zero grain)
```

---

## 🔧 No Configuration Needed

**v1 advantages:**
- ✅ No command-line flags (ZERO - not permitted by design)
- ✅ Must run via START_HERE.bat or directly without arguments
- ✅ All settings built-in and cannot be modified
- ✅ No parallel processing to configure
- ✅ No upscaling parameters to tune
- ✅ Auto-detects GPU (uses if available)
- ✅ Fixed paths (works from any directory)
- ✅ Sequential processing (no memory thrashing)

**⚠️ CLI FLAG ENFORCEMENT:**
The script actively rejects any command-line arguments. If you try to pass flags, it will display an error and exit. This is by design to ensure consistent, reliable operation.

Just put images in `scans/` and run via BAT file. That's it.

---

## 🚫 What NOT to Do

❌ Don't use older versions (v2, v3, v4) - they have inferior quality  
❌ Don't manually edit the script - it's production-ready as-is  
❌ Don't pass command-line flags - **FORBIDDEN: script will reject them**  
❌ Don't run with arguments - run via START_HERE.bat or without args only  
❌ Don't upscale images first - v1 works at native resolution  
❌ Don't use JPEG output - PNG is required for exact colors  

**⚠️ CLI FLAG RULE:** The script enforces zero-configuration operation. Any attempt to pass CLI arguments will be rejected with an error message.  

---

## 📈 Performance

**Typical image (1717x1764):**
- Processing time: 5-10 seconds (CPU)
- Processing time: 2-5 seconds (GPU with CuPy)
- Output size: ~240KB PNG
- Memory usage: <1GB per image

**Large image (4000x5000):**
- Processing time: 30-60 seconds (CPU)
- Processing time: 10-20 seconds (GPU)
- Output size: ~1-2MB PNG
- Memory usage: <2GB per image

---

## 🐛 Troubleshooting

**No output?**
- Check `scans/` folder has images (.jpg, .jpeg, .png, .bmp, .tiff)
- Check `scans/output/` folder was created

**Colors wrong?**
- Verify you're using v1 (`1- newclean-main`)
- Check Python is 3.8+
- Reinstall: `pip install opencv-python numpy --upgrade`

**Script crashes?**
- Image too large (>10,000px) - downscale first
- Not enough RAM - close other programs
- Corrupted image file - try different scan

**Slow processing?**
- Install CuPy for GPU: `pip install cupy-cuda12x`
- Reduce image resolution before processing
- Use SSD instead of HDD for I/O

---

## 🎓 Technical Details

**Why HLS instead of HSV?**
- HLS (Hue-Lightness-Saturation) handles lighting variations better
- Lightness channel (L) more perceptually uniform than HSV Value (V)
- Better color detection under glare and shadows

**Why no upscaling?**
- Upscaling adds processing time and artifacts
- Modern scans are high-res enough (300+ DPI)
- Guided filter + CLAHE preserve detail at native resolution

**Why sequential instead of parallel?**
- Large images use 1-2GB RAM each
- Parallel workers cause memory thrashing and crashes
- Sequential is more reliable and still fast enough

**Why area-open instead of morphological open?**
- Removes noise by connected-component size (not shape)
- Preserves edges of large regions (open erodes uniformly)
- More intelligent than kernel-based operations

---

## 📚 Further Reading

- **[CLEANUP_SCRIPT_PROGRESSION.md](CLEANUP_SCRIPT_PROGRESSION.md)** - Full version comparison
- **[REQUIREMENTS_VERIFICATION.md](REQUIREMENTS_VERIFICATION.md)** - Detailed requirement verification
- **[README.md](README.md)** - Repository overview

---

**Ready to process?** → Put scans in `1- newclean-main/scans/` → Run `START_HERE.bat` → Done! ✨
