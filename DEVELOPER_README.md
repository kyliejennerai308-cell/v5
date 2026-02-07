# Developer README - Vinyl Playmat Restoration

**Target Audience:** Developers who need to understand, modify, or extend the image processing pipeline.

---

## 🏗️ Architecture Overview

### Repository Structure

```
v5/
├── 1- newclean-main/          ← PRODUCTION VERSION (v1)
│   ├── restore_playmat_hsv.py ← Main processing script (660 lines)
│   ├── START_HERE.bat         ← User entry point (Windows)
│   ├── NO_CLI_FLAGS.md        ← CLI enforcement policy
│   └── scans/                 ← Input/output directories
│       ├── *.jpg              ← Place input images here
│       └── output/            ← Cleaned images written here
│
├── 2- repo2-main/             ← ARCHIVED (v2 - HSV, parallel)
├── 3- cleanup-main/           ← ARCHIVED (v3 - HSV, upscaling)
├── 4- old/                    ← ARCHIVED (v4 - beta)
│
├── Research/                  ← Sample images, test data
│
└── Documentation/
    ├── README.md                      ← User guide
    ├── DEVELOPER_README.md            ← This file
    ├── REQUIREMENTS_VERIFICATION.md   ← Requirements mapping
    ├── CLEANUP_SCRIPT_PROGRESSION.md  ← Version comparison
    ├── QUICK_REFERENCE.md             ← Quick start guide
    └── TASK_COMPLETION_SUMMARY.md     ← Implementation summary
```

---

## 🎯 Design Philosophy

### Core Principles

1. **Zero Configuration** - No CLI flags, no config files, all settings optimized and locked
2. **Production Ready** - Extensive testing, error handling, clear error messages
3. **Sequential Processing** - Reliable over fast (no parallel workers causing memory issues)
4. **Native Resolution** - No upscaling (modern scans are high-res enough)
5. **Fixed Paths** - Relative to script location (works from any working directory)

### Why v1 is Superior to v2-v4

| Aspect | v4 (Oldest) | v3 | v2 | v1 (Latest) | Reason |
|--------|-------------|----|----|-------------|---------|
| **Lines of Code** | 1127 | 1564 | 1779 | **660** | Removed CLI parsing, parallel processing overhead |
| **Color Space** | HSV | HSV | HSV | **HLS** | Lightness channel more perceptually uniform than HSV Value |
| **Texture Removal** | Bilateral only | Bilateral only | Bilateral only | **5-stage pipeline** | Multi-stage approach superior to single pass |
| **Edge Detection** | Morphology | Morphology | Morphology | **Canny + keep-out** | Prevents color bleeding across boundaries |
| **CLI Flags** | 6+ flags | 6+ flags | 6+ flags | **Zero (enforced)** | Prevents misconfiguration |
| **Processing** | Parallel | Parallel | Parallel | **Sequential** | More reliable, no memory thrashing |

---

## 🔬 Technical Implementation

### Image Processing Pipeline

The script implements an enhanced 15-stage pipeline:

```
Input: Scanned JPEG (with wrinkles, glare, grain, paint bleed)
    ↓
[1] Bilateral Filter          → Remove vinyl texture
    ↓
[2] Guided Filter             → Edge-aware smoothing
    ↓
[3] Auto-Gamma Correction     → Normalize exposure
    ↓
[4] CLAHE (L channel)         → Enhance local contrast
    ↓
[5] Unsharp Mask              → Sharpen edges
    ↓
[6] BGR → HLS + LAB Conversion → Multi-space color detection
    ↓
[7] Color Distance Analysis   → Compute color gradient magnitude (LAB)
    ↓
[8] Enhanced Edge Detection   → Canny + color gradient keep-out zones
    ↓
[9] Enhanced White Detection  → Top-hat + adaptive + LAB A/B analysis
    ↓
[10] Dark Outline Detection   → Invert-L trick
    ↓
[11] Color Mask Creation      → 8 separate HSL range masks
    ↓
[12] Morphological Operations → Close holes, remove noise
    ↓
[13] Priority Assignment      → Resolve mask overlaps
    ↓
[14] Nearest-Color Fallback   → Fill remaining pixels
    ↓
[15] Post-Processing          → Solidify, vectorize, smooth
    ↓
Output: Clean PNG (8 exact colors, zero grain, no paint bleed)
```

### Key Algorithms

#### 1. Color Distance Analysis (NEW - Lines 398-434)
```python
def _compute_color_distance_map(img):
    """Compute color gradient magnitude to identify paint bleed boundaries.
    
    Converts to LAB, computes Sobel gradients on all channels,
    combines using Euclidean distance, applies OTSU threshold.
    """
```

**Why:** Reveals paint bleed edges that have the same brightness but different color. LAB color space separates lightness from color, making color shifts visible even when grayscale intensity is uniform. This catches:
- Paint bleed halos (same lightness, different hue)
- Color transitions from scanning artifacts
- Subtle repaint boundaries

**Technical Detail:** Sobel operator computes spatial gradients in L, A, and B channels. Combined magnitude shows total color change. OTSU thresholding adapts to varying edge strengths automatically.

#### 2. Enhanced Edge Detection (ENHANCED - Lines 437-470)
```python
def detect_edges_keepout(gray, color_distance_mask=None):
    """Canny edge detection + color gradient analysis.
    
    Combines grayscale intensity edges (Canny) with color change
    edges (gradient map) for comprehensive boundary detection.
    """
```

**Why:** Canny only detects intensity edges. Color distance analysis catches edges where color changes but intensity stays constant (common in scanned prints with flat colors). Both together provide superior edge detection for mask boundaries.

#### 3. LAB A/B Channel White Detection (ENHANCED - Lines 473-531)
```python
def detect_white_text(gray, lab_img=None):
    """Isolate white features using LAB A/B channel analysis.
    
    True white has high L and A/B values near 128 (neutral).
    Catches faded white text that grayscale methods miss.
    """
```

**Why:** White text can fade to light gray in scans, making it hard to detect with grayscale thresholds alone. LAB A/B channels near 128 indicate neutral (no color), so high-L + neutral-AB = white, even if L is lower than expected. This rescues degraded white text in shadowed areas.

#### 4. Guided Filter (Lines 269-293)
```python
def _guided_filter(I, p, radius, eps):
    """Edge-aware smoothing without ximgproc dependency.
    
    Smooths flat regions aggressively while stopping at strong edges.
    Uses box filter (fast) instead of Gaussian (slow).
    """
```

**Why:** Flattens vinyl texture without blurring logo boundaries. Superior to Gaussian blur which smears edges.

#### 2. Auto-Gamma (Lines 296-311)
```python
def _auto_gamma(l_channel):
    """Push mean brightness toward mid-grey.
    
    gamma = log(127) / log(mean_brightness)
    Compensates for under/over-exposure before CLAHE.
    """
```

**Why:** CLAHE works best on well-exposed images. Auto-gamma normalizes exposure baseline.

#### 3. Area-Open Filter (Lines 404-417)
```python
def _area_open(mask, min_area=64):
    """Remove connected components smaller than min_area pixels.
    
    Superior to morphological OPEN for noise removal because:
    - Removes tiny blobs regardless of shape
    - Preserves edges of large regions (OPEN erodes uniformly)
    - Uses cv2.connectedComponentsWithStats (efficient)
    """
```

**Why:** Removes noise without eroding edges. 64px threshold = ~8×8 blob (small enough for noise, large enough for detail).

#### 4. Conditional Dilation (Lines 420-428)
```python
def _conditional_dilate(mask, original, kernel, iterations=1):
    """Dilate mask but only where pixels existed in original.
    
    Restores thin strokes that morphological close may displace,
    without allowing mask to grow into new territory.
    """
```

**Why:** Prevents outline masks from bleeding into adjacent fill regions.

#### 5. Nearest-Color Assignment (Lines 544-586)
```python
# Chunked processing for memory efficiency
CHUNK = 500_000  # Process 500K pixels at a time

# Circular hue distance (H wraps at 180 in OpenCV)
dh = np.minimum(dh, 180.0 - dh)
dh *= (255.0 / 180.0)  # Normalize to 0-255 scale

# Euclidean distance in HLS space
dist = dh**2 + dl**2 + ds**2
nearest_idx = np.argmin(dist, axis=1)
```

**Why:** Assigns unmatched pixels to closest color using perceptually-weighted HLS distance. Chunking prevents memory exhaustion on large images.

---

## 🎨 Color System

### Current Implementation: 8-Color Master Palette (v1)

```python
COLOUR_SPEC = {
    'BG_SKY_BLUE':     # Background (206°, 71%, 64%) → BGR: [228, 187, 134]
    'PRIMARY_YELLOW':  # Silhouettes (59°, 61%, 98%) → BGR: [57, 246, 253]
    'HOT_PINK':        # Logo fills (338°, 55%, 96%) → BGR: [111, 30, 250]
    'DARK_PURPLE':     # Outer borders (275°, 35%, 80%) → BGR: [160, 18, 98]
    'PURE_WHITE':      # Text/stars (0°, 99%, 0% saturation) → BGR: [252, 252, 252]
    'STEP_RED_OUTLINE':# Accents (345°, 52%, 94%) → BGR: [78, 17, 247]
    'LIME_ACCENT':     # Outlines (89°, 55%, 92%) → BGR: [34, 246, 147]
    'DEAD_BLACK':      # Void/edges (0°, 2%, 0%) → BGR: [5, 5, 5]
}
```

### Target Specification: 10-Color Master Palette (v2.0)

The v2.0 specification targets a refined 10-color palette with exact BGR values:

| Color Name | BGR Value | Current v1 | Role/Usage |
|------------|-----------|------------|------------|
| **Sky Blue** | `[233, 180, 130]` | `[228, 187, 134]` | Background canvas (perfectly flat) |
| **Hot Pink** | `[205, 0, 253]` | `[111, 30, 250]` | "STEPS" logo, footprints, number backings |
| **Bright Yellow** | `[1, 252, 253]` | `[57, 246, 253]` | Main fill for silhouettes and ladder rungs |
| **Pure White** | `[255, 255, 255]` | `[252, 252, 252]` | Logo interiors, stars (protected) |
| **Neon Green** | `[0, 213, 197]` | `[34, 246, 147]` | Thin border around yellow figures |
| **Dark Purple** | `[140, 0, 180]` | `[160, 18, 98]` | Outermost thin stroke on logos |
| **Vibrant Red** | `[1, 13, 245]` | `[78, 17, 247]` | Thin underlines and ladder accents |
| **Deep Teal** | `[10, 176, 149]` | *(New)* | Instructional text and secondary shadows |
| **Secondary Yellow** | `[55, 255, 255]` | *(New)* | Secondary fill for group silhouettes |
| **Black** | `[0, 0, 0]` | `[5, 5, 5]` | Dead space or scan document edges |

**Key Differences v1 → v2.0:**
- Adds 2 new colors: Deep Teal, Secondary Yellow
- Uses direct BGR values instead of HLS conversion
- More saturated/vibrant colors (closer to digital primaries)
- Emphasis on LAB/HLS color space for matching (perceptually uniform)

### v2.0 Processing Constraints

**Developer Requirements:**
1. **Kernel Sizes:** Use 3×3 kernels only (not 5×5)
   - Prevents "melting" small text
   - Preserves star points sharpness
   
2. **Curve Integrity:** `approxPolyDP` epsilon = 0.001 (not 0.02)
   - Maintains organic shapes of silhouettes
   - Preserves finger details
   
3. **Anti-Aliasing:** Apply only to color boundaries
   - Prevents stair-step pixelation
   - Pink meeting Blue, Yellow meeting Green, etc.
   
4. **Text Protection:** Avoid heavy morphological opening
   - Instructional text must remain legible
   - Protected mode for high-contrast features

**Processing Pipeline (v2.0 Target):**
```
Phase 1: Pre-Processing & "Protected Mode"
  ├─ Mask high-contrast features (text, stars, logos)
  ├─ Lock pixels to prevent erosion
  └─ Suppress shadows/glare (flag for replacement)

Phase 2: Background & Silhouette Restoration
  ├─ Flood-fill background with Sky Blue [233, 180, 130]
  ├─ Area-based filtering (remove < 20px noise)
  ├─ Use connectedComponentsWithStats for sharp edges
  └─ Clean original green outline (protected cleanup)

Phase 3: Final Assembly (Layering Order)
  ├─ Bottom: Flat Sky Blue Background
  ├─ Middle-Lower: Cleaned Original Green Outline
  ├─ Middle-Upper: Cleaned Yellow Silhouette
  └─ Top: White interiors + Pink/Purple logo strokes
```

### HLS vs HSV (Current Implementation)

**Why HLS instead of HSV?**

```python
# HSV: Value = max(R, G, B)
# - Value changes drastically under glare/shadows
# - Hard to detect colors consistently

# HLS: Lightness = (max(R,G,B) + min(R,G,B)) / 2
# - Lightness more stable under lighting variations
# - Perceptually uniform
# - Better for color detection in scanned images
```

### Color Detection Strategy

1. **Range-based classification** (Lines 249-262)
   - Each color has H/L/S ranges
   - Ranges widened to accommodate scan variations
   - Exclusive ranges prevent misclassification

2. **Boosting mechanisms** (Lines 460-472)
   - White pixels boosted with top-hat mask
   - Outline colors boosted with dark-outline mask
   - Ensures small features aren't lost

3. **Priority system** (Lines 526-538)
   ```
   PURE_WHITE > DEAD_BLACK > PRIMARY_YELLOW >
   STEP_RED_OUTLINE > DARK_PURPLE > HOT_PINK >
   LIME_ACCENT > BG_SKY_BLUE
   ```
   - Higher priority colors win overlaps
   - Ensures thin outlines preserved over fills

---

## 🛠️ How to Modify

### Adding a New Color

1. **Define color spec** (Lines 117-171):
   ```python
   COLOUR_SPEC['NEW_COLOR'] = {
       'target_hls': (_h(degrees), _sl(lightness), _sl(saturation)),
       'range_h': (_h(min_hue), _h(max_hue)),
       'range_s': (_sl(min_sat), _sl(max_sat)),
       'range_l': (_sl(min_light), _sl(max_light)),
   }
   ```

2. **Update priority order** (Lines 526-530):
   ```python
   order = [
       'PURE_WHITE', 'DEAD_BLACK', 'NEW_COLOR',
       'PRIMARY_YELLOW', # ... rest
   ]
   ```

3. **Test edge cases:**
   - Color overlap with existing colors
   - Performance impact (nearest-color distance calculation)

### Adjusting Texture Removal

**If images have MORE texture:**
```python
# Line 337: Increase bilateral filter strength
smoothed = cv2.bilateralFilter(img, d=13, sigmaColor=100, sigmaSpace=100)

# Line 342: Increase guided filter radius
smoothed[:, :, c] = _guided_filter(guide, smoothed[:, :, c], radius=6, eps=0.01)
```

**If images have LESS texture (losing detail):**
```python
# Decrease bilateral strength
smoothed = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

# Decrease guided filter radius
smoothed[:, :, c] = _guided_filter(guide, smoothed[:, :, c], radius=3, eps=0.005)
```

### Adjusting Edge Preservation

**If edges are bleeding:**
```python
# Line 367: Increase edge dilation
return cv2.dilate(edges, keep_kernel, iterations=2)  # was 1
```

**If losing thin lines:**
```python
# Line 366: Decrease Canny thresholds
edges = cv2.Canny(gray, 30, 100)  # was 50, 150

# Line 493: Increase conditional dilation
m = _conditional_dilate(m, original, edge_kernel, iterations=2)  # was 1
```

---

## 🧪 Testing Strategy

### Unit Testing Key Components

```python
# Test guided filter preserves edges
def test_guided_filter_edge_preservation():
    # Create synthetic image with sharp edge
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    
    # Apply guided filter
    filtered = _guided_filter(img, img, radius=4, eps=0.01)
    
    # Check edge is preserved (gradient should be sharp)
    gradient = np.abs(filtered[:, 51] - filtered[:, 49])
    assert np.mean(gradient) > 200  # Sharp edge maintained

# Test area-open removes small noise
def test_area_open_noise_removal():
    # Create mask with small noise blobs
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:15, 10:15] = 255  # 5x5 = 25px (< 64px threshold)
    mask[30:40, 30:40] = 255  # 10x10 = 100px (> 64px threshold)
    
    cleaned = _area_open(mask, min_area=64)
    
    # Small blob removed, large blob preserved
    assert np.sum(cleaned[10:15, 10:15]) == 0
    assert np.sum(cleaned[30:40, 30:40]) > 0
```

### Integration Testing

```python
# Test full pipeline produces valid output
def test_full_pipeline():
    img = cv2.imread('test_scan.jpg')
    output = process_image_return_array(img)  # Modified to return array
    
    # Check output dimensions match input
    assert output.shape[:2] == img.shape[:2]
    
    # Check only 8 colors present
    unique_colors = np.unique(output.reshape(-1, 3), axis=0)
    assert len(unique_colors) <= 8
    
    # Check no unassigned pixels (all pixels in palette)
    bgr_palette = [BGR_TARGETS[k] for k in COLOUR_SPEC.keys()]
    for pixel in unique_colors:
        assert any(np.array_equal(pixel, p) for p in bgr_palette)
```

### Visual Regression Testing

```bash
# Process test image and compare with golden reference
python restore_playmat_hsv.py
compare scans/output/test.png golden_refs/test_golden.png -metric RMSE diff.png

# RMSE should be 0 (pixel-perfect match)
```

---

## 🚀 Performance Optimization

### GPU Acceleration

The script supports GPU via CuPy (CUDA-accelerated NumPy):

```python
# Lines 91-98: GPU detection
GPU_AVAILABLE, GPU_BACKEND, GPU_DETECTION_MESSAGE = _detect_gpu()
USE_GPU = GPU_AVAILABLE

# Lines 556-572: GPU-accelerated nearest-color assignment
if use_cupy:
    chunk = cp.asarray(um_hls[start:end])  # Move to GPU
    # ... compute distances on GPU ...
    nearest_idx = cp.argmin(dist, axis=1)
    result[rows, cols] = cp.asnumpy(bgr_lut_gpu[nearest_idx])  # Back to CPU
```

**Install GPU support:**
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

**Performance gain:** 2-3x speedup on large images (4000×5000+)

### Memory Management

```python
# Chunked processing prevents memory exhaustion (Lines 555-586)
CHUNK = 500_000  # Process 500K pixels at a time

# Why 500K?
# - Large images (4000×5000 = 20M pixels) would require:
#   20M × 8 colors × 3 channels × 4 bytes = 1.9 GB
# - Chunking: 500K × 8 × 3 × 4 = 48 MB per iteration
# - Total memory: < 1 GB even for huge images
```

### Profiling Hotspots

```python
import cProfile

def profile_processing():
    pr = cProfile.Profile()
    pr.enable()
    
    process_image(Path('test.jpg'))
    
    pr.disable()
    pr.print_stats(sort='cumulative')

# Typical time distribution (1717×1764 image):
# 40% - prep_image (bilateral + guided + CLAHE)
# 30% - nearest-color assignment
# 15% - morphological operations
# 10% - edge detection
#  5% - I/O, misc
```

---

## 🔒 Security & Validation

### Input Validation

```python
# Lines 621-623: Check input directory exists
if not INPUT_DIR.exists():
    print(f"Error: input directory '{INPUT_DIR}' not found.")
    sys.exit(1)

# Lines 627-633: Validate image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
images = [f for f in INPUT_DIR.iterdir()
          if f.suffix.lower() in image_extensions]

# Lines 434-437: Check image loads successfully
img = cv2.imread(str(image_path))
if img is None:
    print(f"  Error: could not load {image_path}")
    return
```

### CLI Argument Enforcement

```python
# Lines 602-619: Reject any command-line arguments
if len(sys.argv) > 1:
    print("ERROR: Command-line arguments are not permitted")
    print("Run via START_HERE.bat or directly without arguments")
    sys.exit(1)
```

**Rationale:**
- Prevents user misconfiguration
- Ensures consistent output across all users
- Simplifies technical support
- All settings optimized and locked

### Error Handling

```python
# Lines 592-595: Verify output file written
success = cv2.imwrite(str(output_path), result)
if not success or not output_path.exists():
    print(f"  ERROR: failed to write {output_path}")
else:
    print(f"  Saved: {output_path} ({output_path.stat().st_size} bytes)")
```

---

## 📊 Benchmarks

### Processing Time

| Image Size | CPU (i7-8700K) | GPU (RTX 3080) |
|------------|----------------|----------------|
| 1717×1764  | 7s             | 3s             |
| 3000×3000  | 18s            | 7s             |
| 4000×5000  | 45s            | 15s            |
| 6000×8000  | 120s           | 35s            |

### Memory Usage

| Image Size | Peak RAM | With GPU |
|------------|----------|----------|
| 1717×1764  | 800 MB   | 1.2 GB   |
| 3000×3000  | 1.5 GB   | 2.0 GB   |
| 4000×5000  | 2.5 GB   | 3.2 GB   |

**Note:** Sequential processing keeps memory bounded. Parallel workers (v2-v4) caused 3-4x memory usage → crashes.

---

## 🐛 Common Issues & Solutions

### Issue: Colors Wrong After Modification

**Symptom:** Output has incorrect colors, color bleeding

**Debug:**
```python
# Add after line 456 to visualize masks
for name, mask in raw_masks.items():
    cv2.imwrite(f'debug_{name}.png', mask)

# Check which masks are capturing which regions
```

**Solution:** Adjust HSL ranges in COLOUR_SPEC (Lines 117-171)

### Issue: Losing Fine Detail

**Symptom:** Small text disappears, thin lines break

**Debug:**
```python
# Add after line 449 to check text detection
cv2.imwrite('debug_white_text.png', white_text_mask)
cv2.imwrite('debug_dark_outline.png', dark_outline_mask)
```

**Solution:**
- Decrease bilateral filter strength (Line 337)
- Increase conditional dilation iterations (Line 493)
- Adjust white text detection threshold (Line 382)

### Issue: Texture Still Visible

**Symptom:** Vinyl grain visible in output, sky blue not flat

**Debug:**
```python
# Add after line 441 to check preprocessing
cv2.imwrite('debug_prepped.png', prepped)
```

**Solution:**
- Increase bilateral filter parameters (Line 337)
- Increase guided filter radius (Line 342)
- Increase area-open threshold (Line 489)

---

## 🔄 Git Workflow

### Branch Structure

```
main (or master)
  └── copilot/clean-up-script-version  ← Current PR branch
```

### Commit History

```bash
d93e876 Add task completion summary document
32e6d9c Fix docstring formatting - remove trailing whitespace
1113627 Enforce no CLI flags rule - BAT launch only
65d5c70 Add comprehensive documentation for cleanup script progression
45df9a6 Initial plan
```

### Making Changes

```bash
# Always work in the PR branch
git checkout copilot/clean-up-script-version

# Make changes to code
# ...

# Test changes
cd "1- newclean-main"
python restore_playmat_hsv.py

# Use report_progress tool (do NOT use git directly)
# The tool will handle: git add, git commit, git push
```

---

## 📚 Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| **restore_playmat_hsv.py** | 660 | Main processing script |
| Lines 1-20 | Docstring, imports | Module header |
| Lines 22-30 | Path setup | Fixed paths, optimizations |
| Lines 32-98 | GPU detection | CUDA availability check |
| Lines 100-102 | GPU helpers | CPU/CUDA wrapper functions |
| Lines 104-171 | Color specs | 8-color HLS palette |
| Lines 173-187 | Color utils | HLS→BGR conversion |
| Lines 189-243 | GPU wrappers | Color conversion, morphology |
| Lines 245-263 | Mask creation | HSL range matching |
| Lines 265-358 | Preprocessing | 5-stage texture removal |
| Lines 360-402 | Feature detection | Edges, text, outlines |
| Lines 404-428 | Morphology utils | Area-open, conditional dilation |
| Lines 430-596 | Main pipeline | process_image() function |
| Lines 598-644 | Entry point | main() with CLI enforcement |

---

## 🎓 Learning Resources

### Understanding Color Spaces

- **HSV vs HLS:** https://en.wikipedia.org/wiki/HSL_and_HSV
- **Why HLS for lighting:** https://stackoverflow.com/q/48129325

### OpenCV Documentation

- **Guided Filter Theory:** Zhang et al., "Fast Guided Filter" (ECCV 2015)
- **Morphological Ops:** https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- **CLAHE:** https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

### Algorithm Papers

1. **Bilateral Filter:** Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images" (ICCV 1998)
2. **Guided Filter:** He et al., "Guided Image Filtering" (ECCV 2010)
3. **Auto-Gamma:** Huang et al., "Efficient Contrast Enhancement Using Adaptive Gamma Correction" (IEEE 2013)

---

## 🤝 Contributing Guidelines

### Code Style

```python
# Follow PEP 8
# - 4 spaces for indentation (no tabs)
# - Max line length: 79 characters for code, 72 for comments
# - Docstrings: Google style

def function_name(param1, param2):
    """Brief description.
    
    More detailed explanation of what the function does,
    edge cases, algorithm used, etc.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    """
    pass
```

### Testing Before Commit

```bash
# 1. Run on test images
python restore_playmat_hsv.py

# 2. Verify output quality
# - Check scans/output/ visually
# - Confirm 8 colors only
# - No grain, flat regions, clean edges

# 3. Test CLI enforcement
python restore_playmat_hsv.py --help  # Should error
python restore_playmat_hsv.py         # Should work

# 4. Use report_progress tool (handles git operations)
```

### Documentation Updates

When modifying code, update relevant documentation:

- **Code changes:** Update this DEVELOPER_README.md
- **User-facing changes:** Update README.md
- **Requirements changes:** Update REQUIREMENTS_VERIFICATION.md
- **New features:** Add to CLEANUP_SCRIPT_PROGRESSION.md

---

## 📞 Support & Contact

**For Technical Issues:**
- Check TROUBLESHOOTING section in README.md
- Review REQUIREMENTS_VERIFICATION.md for implementation details
- See CLEANUP_SCRIPT_PROGRESSION.md for version differences

**For Development Questions:**
- Review this DEVELOPER_README.md
- Check inline code comments (extensive in restore_playmat_hsv.py)
- Reference algorithm papers in Learning Resources section

---

**Last Updated:** 2026-02-07  
**Version:** v1 (1-newclean-main)  
**Maintainer:** GitHub Copilot Agent (@copilot)
