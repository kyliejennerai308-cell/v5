# Image Analysis and Cleanup Tool

**Purpose:** Analyze scanned images with paint bleed, halos, wrinkles, and texture issues to reveal edges, highlight color shifts, and identify areas for cleanup or vector redraw.

---

## 🎯 Overview

This tool implements advanced computer vision techniques specifically designed for analyzing scanned artwork with:
- **Strong flat colors** (yellow, pink, blue)
- **Paint bleed / halos** around edges
- **Scan texture + wrinkles** from physical material
- **Color shifts** along boundaries
- **Soft transitions** where hard vector edges existed

---

## 🔧 Features

### 1. **LAB Color Space Analysis**
Separates lightness from color information:
- **L channel**: Brightness/lightness
- **A channel**: Green-Red spectrum (great for spotting bleed edges)
- **B channel**: Blue-Yellow spectrum

**Why LAB?** Excellent for identifying paint bleed because color channels are independent of brightness.

### 2. **HSV Color Space Analysis**
Alternative color representation:
- **H channel**: Hue (color type) - shifts pop out around boundaries
- **S channel**: Saturation (color intensity)
- **V channel**: Value (brightness)

**Why HSV?** Hue discontinuities become very visible at paint boundaries.

### 3. **CLAHE Contrast Enhancement**
Contrast Limited Adaptive Histogram Equalization reveals:
- Faint wrinkles in the material
- Subtle color transitions
- Hidden texture from scanning

Applied to:
- LAB L-channel (lightness)
- HSV V-channel (brightness)
- Grayscale image

### 4. **High-Pass Edge Boost**
Enhances paint boundaries using frequency domain filtering:
1. Applies low-pass filter (Gaussian blur)
2. Subtracts from original to isolate high-frequency edges
3. Amplifies and adds back to enhance boundaries

**Result:** Paint edges become more defined and easier to trace.

### 5. **Canny Edge Detection**
Industry-standard edge detection algorithm providing:
- Precise contour detection
- Binary edge map (white edges on black)
- Hysteresis thresholding for clean results

**Use case:** Finding exact boundaries between painted regions.

### 6. **Color Distance Maps**
Shows where colors suddenly change:
- Computes color gradient magnitude using Sobel operator
- Works in both LAB and HSV color spaces
- Higher values indicate rapid color changes (bleed areas)

**Use case:** Identifying repaint areas and color boundaries.

---

## 📦 Installation

### Requirements
- Python 3.6+
- OpenCV (cv2)
- NumPy

### Install Dependencies

```bash
pip install opencv-python numpy
```

Or if you have a requirements file:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Basic Usage

```bash
python image_analysis.py <input_image>
```

Example:
```bash
python image_analysis.py img20260131_17015938.jpg
```

This will:
1. Analyze the image using all techniques
2. Create an output directory named `<image_name>_analysis/`
3. Save all analysis results as PNG files

### Custom Output Directory

```bash
python image_analysis.py <input_image> <output_directory>
```

Example:
```bash
python image_analysis.py scan.jpg my_analysis_results/
```

---

## 📊 Output Files

The script generates multiple analysis images in the output directory:

### Color Space Conversions
- `lab.png` - Full LAB color space image
- `lab_L.png` - LAB Lightness channel
- `lab_A.png` - LAB Green-Red channel
- `lab_B.png` - LAB Blue-Yellow channel
- `hsv.png` - Full HSV color space image
- `hsv_H.png` - HSV Hue channel
- `hsv_S.png` - HSV Saturation channel
- `hsv_V.png` - HSV Value channel

### Contrast Enhancement
- `clahe_lab_L.png` - CLAHE-enhanced LAB L-channel (reveals faint details)
- `clahe_hsv_V.png` - CLAHE-enhanced HSV V-channel
- `clahe_gray.png` - CLAHE-enhanced grayscale

### Edge Analysis
- `edge_boosted.png` - High-pass edge boosted image
- `canny_edges.png` - Binary edge map from Canny detector

### Color Change Detection
- `color_distance_lab.png` - Color distance map (LAB space)
- `color_distance_hsv.png` - Color distance map (HSV space)

### Summary
- `summary_composite.png` - Multi-panel overview of key results

---

## 🎨 Interpreting Results

### For Identifying Paint Bleed

1. **Check `lab_A.png` and `lab_B.png`**
   - Look for halos or gradients around solid colors
   - A/B channels isolate color information without brightness

2. **Examine `color_distance_lab.png` or `color_distance_hsv.png`**
   - Bright areas = rapid color changes
   - Perfect for finding bleed boundaries
   - Use JET colormap for visualization (red = high change)

### For Finding Vector Edges

1. **Review `canny_edges.png`**
   - White lines show detected edges
   - Use as a guide for vector tracing

2. **Check `edge_boosted.png`**
   - Enhanced boundaries make tracing easier
   - Useful for semi-automated vectorization

### For Revealing Hidden Details

1. **Examine `clahe_lab_L.png`**
   - Shows wrinkles and texture normally invisible
   - Helps identify areas needing cleanup

2. **Check `hsv_H.png` with colormap**
   - Hue discontinuities reveal subtle color shifts
   - Great for finding inconsistent coloring

### For Cleanup Preparation

1. **Compare Original with `edge_boosted.png`**
   - Decide which edges are real vs. texture
   - Plan cleanup strategy

2. **Use `color_distance_*.png` maps**
   - Identify problem areas requiring attention
   - Prioritize cleanup efforts

---

## 💡 Advanced Usage

### Using as a Python Module

```python
from image_analysis import ImageAnalyzer

# Create analyzer
analyzer = ImageAnalyzer("scan.jpg")

# Run individual analyses
analyzer.convert_to_lab()
analyzer.apply_clahe(clip_limit=3.0)  # More aggressive
analyzer.canny_edge_detection(threshold1=100, threshold2=200)  # Stricter

# Or run everything
analyzer.analyze_all()

# Save results
analyzer.save_results("custom_output/")

# Access results directly
lab_image = analyzer.results['lab']
edges = analyzer.results['canny_edges']
```

### Customizing Parameters

#### CLAHE Enhancement
```python
analyzer.apply_clahe(
    clip_limit=3.0,      # Higher = more contrast (default: 2.0)
    tile_size=(16, 16)   # Larger = smoother (default: (8, 8))
)
```

#### High-Pass Edge Boost
```python
analyzer.high_pass_edge_boost(
    kernel_size=7,    # Larger = softer edges (default: 5)
    sigma=2.0,        # Blur strength (default: 1.5)
    amount=2.0        # Edge amplification (default: 1.5)
)
```

#### Canny Edge Detection
```python
analyzer.canny_edge_detection(
    threshold1=30,    # Lower = more edges (default: 50)
    threshold2=100    # Upper threshold (default: 150)
)
```

---

## 🔬 Technical Details

### Color Space Selection

**LAB (Recommended for paint bleed)**
- Perceptually uniform color space
- Separates lightness from color
- A/B channels: opponent color dimensions
- Best for identifying subtle color shifts

**HSV (Recommended for vector edge finding)**
- Intuitive color representation
- Hue channel shows pure color boundaries
- Good for segmenting colored regions

### CLAHE Algorithm

Contrast Limited Adaptive Histogram Equalization:
- Divides image into tiles (default: 8x8)
- Applies histogram equalization to each tile
- Limits contrast to prevent noise amplification
- Excellent for revealing hidden details

### High-Pass Filtering

Frequency domain technique:
- Low frequencies = smooth gradients, large shapes
- High frequencies = edges, texture, details
- Subtracting blur isolates high frequencies
- Adding back creates sharpened result

### Canny Edge Detection

Multi-stage algorithm:
1. Gaussian blur (noise reduction)
2. Gradient calculation (Sobel operators)
3. Non-maximum suppression (thin edges)
4. Hysteresis thresholding (strong/weak edges)

### Color Distance Maps

Gradient magnitude computation:
- Sobel operator finds color gradients
- Computed in all 3 channels of color space
- Combined using Euclidean distance
- Normalized to 0-255 for visualization

---

## 🎓 Example Workflow

### Preparing Scans for Vector Redraw

1. **Run Analysis**
   ```bash
   python image_analysis.py artwork_scan.jpg
   ```

2. **Review Edge Detection**
   - Open `canny_edges.png` in image editor
   - Use as tracing guide layer

3. **Identify Problem Areas**
   - Check `color_distance_lab.png`
   - Red/yellow areas = color bleeding
   - Mark for manual cleanup

4. **Check Hidden Issues**
   - View `clahe_lab_L.png`
   - Note wrinkles and texture
   - Plan cleanup strategy

5. **Export for Vector Software**
   - Use `edge_boosted.png` for auto-tracing
   - Reference `hsv_H.png` for color selection

### Cleaning Scanned Prints

1. **Analyze First**
   ```bash
   python image_analysis.py print_scan.jpg cleanup_analysis/
   ```

2. **Identify Bleed**
   - Open `color_distance_lab.png`
   - High-intensity areas = bleed/halos
   - Target for correction

3. **Find True Edges**
   - Compare `canny_edges.png` vs. original
   - True edges = consistent across channels
   - Texture edges = only in lightness

4. **Color Correction Planning**
   - Review `hsv_H.png` for hue consistency
   - Check `lab_A.png` and `lab_B.png` for color drift
   - Plan color normalization strategy

---

## 🐛 Troubleshooting

### Issue: "Image not found"
**Solution:** Check the file path and ensure the image exists

### Issue: "Failed to load image"
**Solution:** Ensure the file is a valid image format (JPG, PNG, etc.)

### Issue: Too many/too few edges detected
**Solution:** Adjust Canny thresholds:
```python
# More edges
analyzer.canny_edge_detection(threshold1=30, threshold2=80)

# Fewer edges (cleaner)
analyzer.canny_edge_detection(threshold1=100, threshold2=200)
```

### Issue: CLAHE reveals too much noise
**Solution:** Reduce clip_limit:
```python
analyzer.apply_clahe(clip_limit=1.5)  # More conservative
```

### Issue: Color distance map looks uniform
**Solution:** The image may already be very clean, or colors are similar. Try:
- Viewing with false color (JET colormap)
- Analyzing a specific region with more variation

---

## 📚 Related Documentation

- **README.md** - Main repository overview
- **DEVELOPER_README.md** - Technical implementation details
- **CLEANUP_SCRIPT_PROGRESSION.md** - Evolution of cleanup scripts

---

## 🤝 Integration with Existing Tools

This analysis tool complements the existing restoration scripts:

- **1- newclean-main/restore_playmat_hsv.py** - Automated cleanup
- **image_analysis.py** - Manual inspection and planning

**Workflow:**
1. Run `image_analysis.py` to understand the scan
2. Use results to plan cleanup strategy
3. Run `restore_playmat_hsv.py` for automated processing
4. Manually address areas identified by analysis

---

## 📝 Notes

- All outputs are PNG format for lossless quality
- Color distance maps use grayscale (0-255)
- Summary composite includes labels and colormaps
- Original image is never modified
- Safe to run multiple times (overwrites previous results)

---

## 🎯 Best Practices

1. **Always start with analysis before cleanup**
2. **Compare multiple color spaces** (LAB vs. HSV)
3. **Use CLAHE results to find hidden issues**
4. **Color distance maps identify priorities**
5. **Canny edges guide vector tracing**
6. **Keep original scans for reference**

---

## ✅ Quick Reference

| Task | Recommended Output |
|------|-------------------|
| Find paint bleed | `color_distance_lab.png` |
| Identify vector edges | `canny_edges.png` |
| Reveal hidden texture | `clahe_lab_L.png` |
| Check color consistency | `hsv_H.png` |
| Plan cleanup areas | `summary_composite.png` |
| Prepare for tracing | `edge_boosted.png` |

---

**Ready to analyze?**
```bash
python image_analysis.py your_scan.jpg
```
