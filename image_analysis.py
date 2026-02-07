#!/usr/bin/env python3
"""
Image Analysis and Cleanup Script
==================================

This script analyzes scanned images with paint bleed, halos, wrinkles, and texture issues.
It provides tools for revealing edges, highlighting color shifts, and isolating paint bleed.

Core Techniques:
- LAB and HSV color space conversion (separates lightness from color)
- CLAHE contrast enhancement (reveals faint wrinkles and transitions)
- High-pass edge boost (enhances paint boundaries)
- Canny edge detection (precise contour detection)
- Color distance maps (identifies areas where color suddenly changes)

Usage:
    python image_analysis.py <input_image> [output_directory]

Examples:
    python image_analysis.py scan.jpg
    python image_analysis.py scan.jpg analysis_output/
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


class ImageAnalyzer:
    """Analyzes scanned images to reveal edges, color shifts, and paint bleed."""
    
    def __init__(self, image_path):
        """
        Initialize the analyzer with an input image.
        
        Args:
            image_path: Path to the input image file
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image in BGR format (OpenCV default)
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        print(f"✓ Loaded image: {self.image_path.name}")
        print(f"  Resolution: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        
        # Storage for analysis results
        self.results = {}
    
    def convert_to_lab(self):
        """
        Convert image to LAB color space.
        
        LAB separates:
        - L channel: Lightness (brightness)
        - A channel: Green-Red color spectrum
        - B channel: Blue-Yellow color spectrum
        
        This is ideal for spotting paint bleed edges.
        
        Returns:
            LAB image (numpy array)
        """
        print("\n[1/5] Converting to LAB color space...")
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Split into separate channels for analysis
        L, A, B = cv2.split(lab)
        
        self.results['lab'] = lab
        self.results['lab_L'] = L
        self.results['lab_A'] = A
        self.results['lab_B'] = B
        
        print("  ✓ LAB conversion complete")
        print(f"    L channel range: {L.min()}-{L.max()}")
        print(f"    A channel range: {A.min()}-{A.max()}")
        print(f"    B channel range: {B.min()}-{B.max()}")
        
        return lab
    
    def convert_to_hsv(self):
        """
        Convert image to HSV color space.
        
        HSV separates:
        - H channel: Hue (color type)
        - S channel: Saturation (color intensity)
        - V channel: Value (brightness)
        
        Hue shifts around boundaries become very visible.
        
        Returns:
            HSV image (numpy array)
        """
        print("\n[2/5] Converting to HSV color space...")
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Split into separate channels
        H, S, V = cv2.split(hsv)
        
        self.results['hsv'] = hsv
        self.results['hsv_H'] = H
        self.results['hsv_S'] = S
        self.results['hsv_V'] = V
        
        print("  ✓ HSV conversion complete")
        print(f"    H channel range: {H.min()}-{H.max()}")
        print(f"    S channel range: {S.min()}-{S.max()}")
        print(f"    V channel range: {V.min()}-{V.max()}")
        
        return hsv
    
    def apply_clahe(self, clip_limit=2.0, tile_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        This reveals faint wrinkles, transitions, and texture that's hard to see.
        Works best on the L channel of LAB or V channel of HSV.
        
        Args:
            clip_limit: Contrast clipping threshold (higher = more contrast)
            tile_size: Size of grid for local histogram equalization
        
        Returns:
            Dictionary with CLAHE-enhanced images
        """
        print("\n[3/5] Applying CLAHE contrast enhancement...")
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        # Apply CLAHE to LAB L-channel
        if 'lab_L' in self.results:
            lab_L_clahe = clahe.apply(self.results['lab_L'])
            self.results['clahe_lab_L'] = lab_L_clahe
            print(f"  ✓ CLAHE applied to LAB L-channel")
        
        # Apply CLAHE to HSV V-channel
        if 'hsv_V' in self.results:
            hsv_V_clahe = clahe.apply(self.results['hsv_V'])
            self.results['clahe_hsv_V'] = hsv_V_clahe
            print(f"  ✓ CLAHE applied to HSV V-channel")
        
        # Apply to grayscale version for general contrast enhancement
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe_gray = clahe.apply(gray)
        self.results['clahe_gray'] = clahe_gray
        print(f"  ✓ CLAHE applied to grayscale")
        
        return self.results
    
    def high_pass_edge_boost(self, kernel_size=5, sigma=1.5, amount=1.5):
        """
        Apply high-pass filter to enhance paint boundaries.
        
        This technique:
        1. Blurs the image (low-pass filter)
        2. Subtracts blur from original (high-pass = edges)
        3. Adds edges back to enhance boundaries
        
        Args:
            kernel_size: Size of Gaussian blur kernel (odd number)
            sigma: Gaussian blur strength
            amount: How much to boost edges (1.0 = normal, 2.0 = double)
        
        Returns:
            Edge-boosted image
        """
        print("\n[4/5] Applying high-pass edge boost...")
        
        # Convert to float for precise arithmetic
        img_float = self.image.astype(np.float32)
        
        # Low-pass filter (blur)
        blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
        
        # High-pass = original - blurred (edges only)
        high_pass = img_float - blurred
        
        # Boost edges and add back to original
        edge_boosted = img_float + (high_pass * amount)
        
        # Clip to valid range and convert back
        edge_boosted = np.clip(edge_boosted, 0, 255).astype(np.uint8)
        
        self.results['edge_boosted'] = edge_boosted
        print(f"  ✓ Edge boost complete (amount={amount})")
        
        return edge_boosted
    
    def canny_edge_detection(self, threshold1=50, threshold2=150):
        """
        Apply Canny edge detection for precise contour detection.
        
        This finds exact boundaries between painted regions.
        
        Args:
            threshold1: Lower threshold for hysteresis
            threshold2: Upper threshold for hysteresis
        
        Returns:
            Binary edge map (white edges on black background)
        """
        print("\n[5/5] Applying Canny edge detection...")
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
        self.results['canny_edges'] = edges
        
        # Count edge pixels
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        print(f"  ✓ Edge detection complete")
        print(f"    Edge pixels: {edge_pixels:,} ({edge_percentage:.2f}%)")
        
        return edges
    
    def color_distance_map(self, method='lab'):
        """
        Generate color distance map showing where colors suddenly change.
        
        This is ideal for identifying paint bleed and repaint areas.
        Uses the gradient magnitude of color channels.
        
        Args:
            method: Color space to use ('lab' or 'hsv')
        
        Returns:
            Color distance map (grayscale, higher values = more color change)
        """
        print("\n[BONUS] Computing color distance map...")
        
        if method == 'lab' and 'lab' in self.results:
            color_image = self.results['lab']
            print("  Using LAB color space")
        elif method == 'hsv' and 'hsv' in self.results:
            color_image = self.results['hsv']
            print("  Using HSV color space")
        else:
            print("  Warning: Color space not available, converting...")
            if method == 'lab':
                color_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            else:
                color_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Split channels
        ch1, ch2, ch3 = cv2.split(color_image)
        
        # Compute gradient magnitude for each channel using Sobel operator
        grad_ch1_x = cv2.Sobel(ch1, cv2.CV_64F, 1, 0, ksize=3)
        grad_ch1_y = cv2.Sobel(ch1, cv2.CV_64F, 0, 1, ksize=3)
        grad_ch1 = np.sqrt(grad_ch1_x**2 + grad_ch1_y**2)
        
        grad_ch2_x = cv2.Sobel(ch2, cv2.CV_64F, 1, 0, ksize=3)
        grad_ch2_y = cv2.Sobel(ch2, cv2.CV_64F, 0, 1, ksize=3)
        grad_ch2 = np.sqrt(grad_ch2_x**2 + grad_ch2_y**2)
        
        grad_ch3_x = cv2.Sobel(ch3, cv2.CV_64F, 1, 0, ksize=3)
        grad_ch3_y = cv2.Sobel(ch3, cv2.CV_64F, 0, 1, ksize=3)
        grad_ch3 = np.sqrt(grad_ch3_x**2 + grad_ch3_y**2)
        
        # Combine gradients (color distance = total color change)
        color_distance = np.sqrt(grad_ch1**2 + grad_ch2**2 + grad_ch3**2)
        
        # Normalize to 0-255 range
        color_distance = cv2.normalize(color_distance, None, 0, 255, cv2.NORM_MINMAX)
        color_distance = color_distance.astype(np.uint8)
        
        self.results[f'color_distance_{method}'] = color_distance
        
        # Statistics
        mean_distance = np.mean(color_distance)
        max_distance = np.max(color_distance)
        
        print(f"  ✓ Color distance map complete")
        print(f"    Mean distance: {mean_distance:.2f}")
        print(f"    Max distance: {max_distance:.2f}")
        
        return color_distance
    
    def analyze_all(self):
        """
        Run all analysis techniques in sequence.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("\n" + "="*60)
        print("IMAGE ANALYSIS PIPELINE")
        print("="*60)
        
        # Run all analysis steps
        self.convert_to_lab()
        self.convert_to_hsv()
        self.apply_clahe()
        self.high_pass_edge_boost()
        self.canny_edge_detection()
        
        # Generate color distance maps
        self.color_distance_map(method='lab')
        self.color_distance_map(method='hsv')
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return self.results
    
    def save_results(self, output_dir=None):
        """
        Save all analysis results to disk.
        
        Args:
            output_dir: Directory to save results (default: <image_name>_analysis/)
        """
        if not self.results:
            print("No results to save. Run analyze_all() first.")
            return
        
        # Create output directory
        if output_dir is None:
            output_dir = self.image_path.stem + "_analysis"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📁 Saving results to: {output_path}/")
        
        saved_count = 0
        
        # Save all results
        for name, data in self.results.items():
            filename = output_path / f"{name}.png"
            cv2.imwrite(str(filename), data)
            saved_count += 1
            print(f"  ✓ Saved: {filename.name}")
        
        # Create a summary composite image showing key results
        self._create_summary_composite(output_path)
        
        print(f"\n✓ Saved {saved_count} analysis images")
        print(f"✓ Output directory: {output_path.absolute()}")
        
        return output_path
    
    def _create_summary_composite(self, output_path):
        """Create a multi-panel summary image showing key analysis results."""
        
        # Select key results to display
        key_results = []
        labels = []
        
        # Original
        if self.image is not None:
            key_results.append(self.image)
            labels.append("Original")
        
        # LAB L-channel with CLAHE
        if 'clahe_lab_L' in self.results:
            # Convert to BGR for display
            lab_L_bgr = cv2.cvtColor(self.results['clahe_lab_L'], cv2.COLOR_GRAY2BGR)
            key_results.append(lab_L_bgr)
            labels.append("LAB L-channel (CLAHE)")
        
        # Edge boosted
        if 'edge_boosted' in self.results:
            key_results.append(self.results['edge_boosted'])
            labels.append("Edge Boosted")
        
        # Canny edges
        if 'canny_edges' in self.results:
            edges_bgr = cv2.cvtColor(self.results['canny_edges'], cv2.COLOR_GRAY2BGR)
            key_results.append(edges_bgr)
            labels.append("Canny Edges")
        
        # Color distance map
        if 'color_distance_lab' in self.results:
            # Apply colormap for better visualization
            color_dist_colored = cv2.applyColorMap(self.results['color_distance_lab'], cv2.COLORMAP_JET)
            key_results.append(color_dist_colored)
            labels.append("Color Distance Map (LAB)")
        
        # HSV Hue channel
        if 'hsv_H' in self.results:
            hue_colored = cv2.applyColorMap(self.results['hsv_H'], cv2.COLORMAP_HSV)
            key_results.append(hue_colored)
            labels.append("HSV Hue Channel")
        
        if not key_results:
            return
        
        # Resize all images to same height for tiling
        target_height = 800
        resized = []
        for img in key_results:
            h, w = img.shape[:2]
            aspect = w / h
            new_w = int(target_height * aspect)
            resized_img = cv2.resize(img, (new_w, target_height))
            
            # Add label
            labeled_img = resized_img.copy()
            cv2.putText(labeled_img, labels[len(resized)], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(labeled_img, labels[len(resized)], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            resized.append(labeled_img)
        
        # Create grid layout (2 rows)
        if len(resized) <= 3:
            # Single row
            composite = np.hstack(resized)
        else:
            # Two rows
            mid = (len(resized) + 1) // 2
            row1 = np.hstack(resized[:mid])
            row2 = np.hstack(resized[mid:])
            
            # Pad shorter row to match width
            if row1.shape[1] > row2.shape[1]:
                padding = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
                row2 = np.hstack([row2, padding])
            elif row2.shape[1] > row1.shape[1]:
                padding = np.zeros((row1.shape[0], row2.shape[1] - row1.shape[1], 3), dtype=np.uint8)
                row1 = np.hstack([row1, padding])
            
            composite = np.vstack([row1, row2])
        
        # Save composite
        composite_path = output_path / "summary_composite.png"
        cv2.imwrite(str(composite_path), composite)
        print(f"  ✓ Saved: summary_composite.png")


def main():
    """Main entry point for the script."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python image_analysis.py <input_image> [output_directory]")
        print("\nExamples:")
        print("  python image_analysis.py scan.jpg")
        print("  python image_analysis.py scan.jpg analysis_output/")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Create analyzer
        analyzer = ImageAnalyzer(input_image)
        
        # Run all analysis
        analyzer.analyze_all()
        
        # Save results
        analyzer.save_results(output_dir)
        
        print("\n✅ SUCCESS: Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
