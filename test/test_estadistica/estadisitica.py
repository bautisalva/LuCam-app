# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:03:36 2025

@author: Marina
"""

import os
import numpy as np
import datetime
from lucam import Lucam, API


camera = Lucam()
camera.CameraClose()
camera = Lucam()
frameformat, fps = camera.GetFormat()
frameformat.pixelFormat = API.LUCAM_PF_16
camera.SetFormat(frameformat, fps)
print("[INFO] Lucam camera initialized.")

# Set up output directory
output_dir = r'C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica'
os.makedirs(output_dir, exist_ok=True)

# Number of images
N = 1000

print(f"[INFO] Capturing {N} 16-bit images and saving as float matrices...")

for i in range(N):
    img = camera.TakeSnapshot()
    if img is None:
        print(f"[WARNING] Snapshot {i} returned None. Skipping.")
        continue

    img_float = img.astype(np.float32)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(output_dir, f"snapshot_{i:04d}_{timestamp}.npy")
    np.save(filename, img_float)

    if (i + 1) % 100 == 0:
        print(f"[INFO] Saved {i+1} snapshots...")

print("[DONE] All snapshots saved.")

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
from numpy.fft import fft2, fftshift
from datetime import datetime

# === Configuration ===
input_folder = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica"
output_folder = os.path.join(input_folder, "analysis_output")
os.makedirs(output_folder, exist_ok=True)
paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
num_images = min(1000, len(paths))  # up to 1000 images

# === Initialize containers ===
means, stds, cvs = [], [], []
mean_image = None
first_image = None

# === Process images one by one (no stacking!) ===
for i, path in enumerate(paths[:num_images]):
    img = np.load(path)
    if first_image is None:
        first_image = img.copy()

    m = np.mean(img)
    s = np.std(img)
    means.append(m)
    stds.append(s)
    cvs.append(s / m if m > 0 else 0)

    if mean_image is None:
        mean_image = img.astype(np.float64)
    else:
        mean_image += img

mean_image /= num_images
global_mean = np.mean(mean_image)
fpn = mean_image - global_mean
fpn_std = np.std(fpn)

# === Detect line artifacts via row-wise averaging ===
row_profiles = []  # collect per-image row mean

for path in paths[:num_images]:
    img = np.load(path)
    row_mean = np.mean(img, axis=1)  # mean of each row
    row_profiles.append(row_mean)

row_profiles = np.array(row_profiles)  # shape: (num_images, num_rows)
mean_row_profile = np.mean(row_profiles, axis=0)  # average pattern across all images
row_var_profile = np.var(row_profiles, axis=0)  # variation of each row's mean across images

# Plot mean row profile and variance
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(mean_row_profile)
plt.title("Average Row Profile Across Images")

plt.subplot(2, 1, 2)
plt.plot(row_var_profile)
plt.title("Variance of Row Means Across Images")
plt.xlabel("Row Index")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_artifact_analysis.png"))
plt.close()

# === Attempt Line Removal (on first image only for preview) ===
corrected = first_image - mean_row_profile[:, np.newaxis]

# Clip to valid range and cast
corrected = np.clip(corrected, 0, None).astype(np.uint16)

# Save comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(first_image, cmap='gray', vmin=np.percentile(first_image, 1), vmax=np.percentile(first_image, 99))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(corrected, cmap='gray', vmin=np.percentile(corrected, 1), vmax=np.percentile(corrected, 99))
plt.title("After Line Subtraction")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "corrected_vs_original.png"))
plt.close()

# === Enhanced Line Artifact Removal ===
def remove_horizontal_artifacts(image, row_profile):
    """Remove horizontal artifacts by subtracting the average row pattern"""
    # Subtract row pattern (broadcast to full image)
    corrected = image - row_profile[:, np.newaxis]
    
    # Preserve original statistics by adding back the global mean
    original_mean = np.mean(image)
    corrected_mean = np.mean(corrected)
    corrected = corrected + (original_mean - corrected_mean)
    
    return corrected

# Apply to all images and save corrected versions
corrected_dir = os.path.join(output_folder, "corrected_images")
os.makedirs(corrected_dir, exist_ok=True)

for i, path in enumerate(paths[:num_images]):
    img = np.load(path)
    corrected_img = remove_horizontal_artifacts(img, mean_row_profile)
    
    # Save corrected image
    filename = os.path.join(corrected_dir, f"corrected_{i:04d}.npy")
    np.save(filename, corrected_img)
    
    # Save preview PNG for first few images
    if i < 5:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        plt.title("Original")
        
        plt.subplot(1, 2, 2)
        plt.imshow(corrected_img, cmap='gray', vmin=np.percentile(corrected_img, 1), vmax=np.percentile(corrected_img, 99))
        plt.title("Corrected")
        plt.tight_layout()
        plt.savefig(os.path.join(corrected_dir, f"comparison_{i:04d}.png"))
        plt.close()
        
        
# === Plot 1: Mean / Std / CV over time ===
plt.figure()
plt.plot(means, label="Mean")
plt.plot(stds, label="Std")
plt.plot(cvs, label="Coeff. of Variation")
plt.title("Mean / Std / CV over Time")
plt.xlabel("Frame Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "mean_std_cv_over_time.png"))
plt.close()

# === Plot 2: Histograms of first 5 images ===
for i, path in enumerate(paths[:1]):
    img = np.load(path)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    plt.title(f"Snapshot {i}")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), bins=512, range=(np.min(img), np.max(img)))
    plt.title("Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"snapshot_{i:02d}_plot.png"))
    plt.close()

# === Plot 3: Row and Column Profiles of first image ===
row_profile = np.mean(first_image, axis=1)
col_profile = np.mean(first_image, axis=0)
row_std = np.std(row_profile)
col_std = np.std(col_profile)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(row_profile)
plt.title(f"Row Profile (std={row_std:.2f})")

plt.subplot(1, 2, 2)
plt.plot(col_profile)
plt.title(f"Column Profile (std={col_std:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_col_profiles.png"))
plt.close()

# === Plot 4: Fixed Pattern Noise ===
plt.figure()
plt.imshow(fpn, cmap='seismic', vmin=-100, vmax=100)
plt.title(f"Fixed Pattern Noise (std = {fpn_std:.2f})")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fpn_map.png"))
plt.close()

# === Plot 5: FFT Spectrum of first image ===
img_fft = first_image - np.mean(first_image)
f = fft2(img_fft)
fshift = fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure()
plt.imshow(magnitude_spectrum, cmap='inferno')
plt.title("FFT Magnitude Spectrum (1st Image)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fft_spectrum.png"))
plt.close()

# === Plot: Row Profile Variability ===
plt.figure(figsize=(10, 6))
for i in range(min(20, num_images)):  # plot first 20 images' row profiles
    plt.plot(row_profiles[i], alpha=0.2, color='blue')
plt.plot(mean_row_profile, color='red', linewidth=2, label='Mean Profile')
plt.title("Row Profiles (First 20 Images)")
plt.xlabel("Row Index")
plt.ylabel("Row Mean Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_profile_variability.png"))
plt.close()

# === Plot: Row Profile Differences ===
plt.figure(figsize=(10, 4))
plt.plot(mean_row_profile - np.mean(mean_row_profile))
plt.title("Mean Row Profile (Centered)")
plt.xlabel("Row Index")
plt.ylabel("Deviation from Global Mean")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_profile_deviation.png"))
plt.close()
# === Save Summary CSV ===
summary_csv_path = os.path.join(output_folder, "summary_metrics.csv")
with open(summary_csv_path, mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["# Images", num_images])
    writer.writerow(["Image Shape", f"{first_image.shape[0]}x{first_image.shape[1]}"])
    writer.writerow(["Global Mean", f"{global_mean:.2f}"])
    writer.writerow(["FPN Std", f"{fpn_std:.2f}"])
    writer.writerow(["Row Profile Std", f"{row_std:.2f}"])
    writer.writerow(["Column Profile Std", f"{col_std:.2f}"])
    writer.writerow(["Mean of CVs", f"{np.mean(cvs):.4f}"])
    writer.writerow(["Max CV", f"{np.max(cvs):.4f}"])
    writer.writerow(["Min CV", f"{np.min(cvs):.4f}"])

print(f"[DONE] Analysis complete. Results saved to:\n{output_folder}")
#%%

# -*- coding: utf-8 -*-
"""
Enhanced Camera Artifact Analysis Tool
Created on Mon May 12 13:03:36 2025
@author: Marina
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
from numpy.fft import fft2, fftshift
from scipy import ndimage
from datetime import datetime

# ========================
# CONFIGURATION SECTION
# ========================
input_folder = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica"
output_folder = os.path.join(input_folder, "analysis_output")
os.makedirs(output_folder, exist_ok=True)

# Processing parameters
num_images = 1000  # Maximum number of images to process
preview_images = 5  # Number of images to save previews for

# ========================
# DATA LOADING SECTION
# ========================
print("[INFO] Loading image data...")
paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
num_images = min(num_images, len(paths))
print(f"[INFO] Found {len(paths)} images, processing {num_images}")

# ========================
# ANALYSIS FUNCTIONS
# ========================
def remove_horizontal_artifacts(image, row_profile):
    """Enhanced artifact removal with statistics preservation"""
    # Subtract row pattern while maintaining global statistics
    corrected = image - row_profile[:, np.newaxis]
    corrected += np.mean(image) - np.mean(corrected)
    return corrected

def analyze_row_artifacts(row_profiles):
    """Quantify row artifact characteristics"""
    mean_profile = np.mean(row_profiles, axis=0)
    centered_profile = mean_profile - np.mean(mean_profile)
    artifact_strength = np.max(np.abs(centered_profile))
    return mean_profile, centered_profile, artifact_strength

# ========================
# MAIN PROCESSING PIPELINE
# ========================
# Initialize data containers
means, stds, cvs = [], [], []
all_images = []
row_profiles = []

print("[INFO] Processing images and calculating statistics...")
for i, path in enumerate(paths[:num_images]):
    # Load image
    img = np.load(path)
    all_images.append(img)
    
    # Basic statistics
    m = np.mean(img)
    s = np.std(img)
    means.append(m)
    stds.append(s)
    cvs.append(s / m if m > 0 else 0)
    
    # Row profile analysis
    row_mean = np.mean(img, axis=1)
    row_profiles.append(row_mean)
    
    # Progress reporting
    if (i + 1) % 100 == 0:
        print(f"[PROGRESS] Processed {i+1} images...")

# Convert to arrays for vectorized operations
row_profiles = np.array(row_profiles)
all_images = np.array(all_images)

# Calculate composite images
mean_image = np.mean(all_images, axis=0)
global_mean = np.mean(mean_image)
fpn = mean_image - global_mean
fpn_std = np.std(fpn)

# Row artifact analysis
mean_row_profile, centered_row_profile, artifact_strength = analyze_row_artifacts(row_profiles)
row_var_profile = np.var(row_profiles, axis=0)

# ========================
# ARTIFACT CORRECTION
# ========================
print("[INFO] Correcting images for horizontal artifacts...")
corrected_dir = os.path.join(output_folder, "corrected_images")
os.makedirs(corrected_dir, exist_ok=True)

# Calculate correction metrics
original_row_std = np.std(row_profiles, axis=0).mean()
corrected_row_stds = []

for i, img in enumerate(all_images):
    # Apply correction
    corrected_img = remove_horizontal_artifacts(img, mean_row_profile)
    
    # Save corrected image
    np.save(os.path.join(corrected_dir, f"corrected_{i:04d}.npy"), corrected_img)
    
    # Store metrics
    corrected_row_mean = np.mean(corrected_img, axis=1)
    corrected_row_stds.append(np.std(corrected_row_mean))
    
    # Save previews
    if i < preview_images:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        ax1.set_title("Original")
        ax2.imshow(corrected_img, cmap='gray', vmin=np.percentile(corrected_img, 1), vmax=np.percentile(corrected_img, 99))
        ax2.set_title("Corrected")
        plt.tight_layout()
        plt.savefig(os.path.join(corrected_dir, f"comparison_{i:04d}.png"))
        plt.close()

# Calculate improvement metrics
corrected_row_std = np.mean(corrected_row_stds)
improvement_ratio = original_row_std / corrected_row_std

# ========================
# VISUALIZATION SECTION
# ========================
print("[INFO] Generating analysis plots...")

# 1. Basic Statistics Over Time
plt.figure(figsize=(10, 6))
plt.plot(means, label="Mean")
plt.plot(stds, label="Std")
plt.plot(cvs, label="Coeff. of Variation")
plt.title("Image Statistics Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "statistics_over_time.png"))
plt.close()

# 2. Row Artifact Analysis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
ax1.plot(mean_row_profile)
ax1.set_title("Average Row Profile Across Images")
ax1.grid(True)

ax2.plot(row_var_profile)
ax2.set_title("Variance of Row Means Across Images")
ax2.set_xlabel("Row Index")
ax2.grid(True)

ax3.plot(centered_row_profile)
ax3.set_title("Centered Row Profile (Deviation from Mean)")
ax3.set_xlabel("Row Index")
ax3.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_artifact_analysis.png"))
plt.close()

# 3. Row Profile Variability
plt.figure(figsize=(10, 6))
for i in range(min(20, num_images)):
    plt.plot(row_profiles[i], alpha=0.1, color='blue')
plt.plot(mean_row_profile, color='red', linewidth=2, label='Mean Profile')
plt.title(f"Row Profile Variability (First 20 Images)\nArtifact Strength: {artifact_strength:.2f}")
plt.xlabel("Row Index")
plt.ylabel("Row Mean Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_profile_variability.png"))
plt.close()

# 4. Fixed Pattern Noise
plt.figure()
plt.imshow(fpn, cmap='seismic', vmin=-3*fpn_std, vmax=3*fpn_std)
plt.title(f"Fixed Pattern Noise (std = {fpn_std:.2f})")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fpn_map.png"))
plt.close()

# 5. FFT Analysis
img_fft = all_images[0] - np.mean(all_images[0])
f = fft2(img_fft)
fshift = fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure()
plt.imshow(magnitude_spectrum, cmap='inferno')
plt.title("FFT Magnitude Spectrum (1st Image)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fft_spectrum.png"))
plt.close()

# ========================
# SAVE SUMMARY RESULTS
# ========================
print("[INFO] Saving summary metrics...")
summary_csv_path = os.path.join(output_folder, "summary_metrics.csv")
with open(summary_csv_path, mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Number of Images Processed", num_images])
    writer.writerow(["Image Dimensions", f"{all_images[0].shape[0]}x{all_images[0].shape[1]}"])
    writer.writerow(["Global Mean Intensity", f"{global_mean:.2f}"])
    writer.writerow(["Fixed Pattern Noise Std", f"{fpn_std:.2f}"])
    writer.writerow(["Original Row Uniformity (Std)", f"{original_row_std:.2f}"])
    writer.writerow(["Corrected Row Uniformity (Std)", f"{corrected_row_std:.2f}"])
    writer.writerow(["Improvement Ratio", f"{improvement_ratio:.2f}x"])
    writer.writerow(["Horizontal Artifact Strength", f"{artifact_strength:.2f}"])
    writer.writerow(["Mean Coefficient of Variation", f"{np.mean(cvs):.4f}"])
    writer.writerow(["Temporal Noise (Avg Std)", f"{np.mean(stds):.2f}"])

print(f"[DONE] Analysis complete. Results saved to:\n{output_folder}")

#%%

# -*- coding: utf-8 -*-
"""
Memory-Efficient Camera Artifact Analysis
@author: Marina
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
from numpy.fft import fft2, fftshift
from datetime import datetime

# ========================
# CONFIGURATION
# ========================
input_folder = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica"
output_folder = os.path.join(input_folder, "analysis_output")
os.makedirs(output_folder, exist_ok=True)

# Processing parameters
max_images = 1000  # Maximum number of images to process
batch_size = 100   # Process images in batches to save memory
preview_images = 5 # Number of images to save previews for

# ========================
# MEMORY-EFFICIENT FUNCTIONS
# ========================
def process_image_batch(paths):
    """Process images in batches to conserve memory"""
    means, stds, cvs = [], [], []
    row_profiles = []
    sum_image = None
    first_image = None
    
    for i, path in enumerate(paths):
        # Load image as float32 to save memory
        img = np.load(path).astype(np.float32)
        
        if first_image is None:
            first_image = img.copy()
        
        # Calculate basic stats
        m = np.mean(img)
        s = np.std(img, dtype=np.float32)  # Explicit dtype to prevent upcasting
        means.append(m)
        stds.append(s)
        cvs.append(s / m if m > 0 else 0)
        
        # Row profile
        row_profiles.append(np.mean(img, axis=1))
        
        # Accumulate for mean image
        if sum_image is None:
            sum_image = img.astype(np.float64)  # Need float64 for accumulation
        else:
            sum_image += img
        
        # Clear memory
        del img
        
        # Progress reporting
        if (i + 1) % batch_size == 0:
            print(f"[PROGRESS] Processed {i+1} images...")
    
    return {
        'means': means,
        'stds': stds,
        'cvs': cvs,
        'row_profiles': np.array(row_profiles),
        'sum_image': sum_image,
        'first_image': first_image,
        'num_images': len(paths)
    }

# ========================
# MAIN PROCESSING
# ========================
print("[INFO] Locating image files...")
paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
num_images = min(max_images, len(paths))
paths = paths[:num_images]

print(f"[INFO] Processing {num_images} images in memory-efficient mode...")
results = process_image_batch(paths)

# Calculate composite results
mean_image = results['sum_image'] / results['num_images']
global_mean = np.mean(mean_image)
fpn = mean_image - global_mean
fpn_std = np.std(fpn)

# Row artifact analysis
mean_row_profile = np.mean(results['row_profiles'], axis=0)
row_var_profile = np.var(results['row_profiles'], axis=0)

# ========================
# ARTIFACT CORRECTION
# ========================
def correct_artifacts(img, profile):
    """Memory-efficient artifact correction"""
    # Subtract row pattern while maintaining global mean
    corrected = img - profile[:, np.newaxis]
    corrected += np.mean(img) - np.mean(corrected)
    return corrected.astype(np.float32)

print("[INFO] Applying artifact correction...")
corrected_dir = os.path.join(output_folder, "corrected_images")
os.makedirs(corrected_dir, exist_ok=True)

# Calculate original row uniformity
original_row_std = np.std(results['row_profiles'], axis=0).mean()

# Process correction and save
corrected_row_stds = []
for i, path in enumerate(paths):
    img = np.load(path).astype(np.float32)
    corrected = correct_artifacts(img, mean_row_profile)
    
    # Save corrected image
    np.save(os.path.join(corrected_dir, f"corrected_{i:04d}.npy"), corrected)
    
    # Calculate corrected row std
    corrected_row_stds.append(np.std(np.mean(corrected, axis=1)))
    
    # Save previews
    if i < preview_images:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        ax1.set_title("Original")
        ax2.imshow(corrected, cmap='gray', vmin=np.percentile(corrected, 1), vmax=np.percentile(corrected, 99))
        ax2.set_title("Corrected")
        plt.tight_layout()
        plt.savefig(os.path.join(corrected_dir, f"comparison_{i:04d}.png"))
        plt.close()
    
    # Clear memory
    del img, corrected

# Calculate improvement
corrected_row_std = np.mean(corrected_row_stds)
improvement_ratio = original_row_std / corrected_row_std

# ========================
# VISUALIZATION (Memory Optimized)
# ========================
print("[INFO] Generating analysis plots...")

# 1. Basic Statistics
plt.figure(figsize=(10, 6))
plt.plot(results['means'], label="Mean")
plt.plot(results['stds'], label="Std")
plt.plot(results['cvs'], label="CV")
plt.title("Image Statistics Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "statistics_over_time.png"))
plt.close()

# 2. Row Artifact Analysis
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(mean_row_profile)
plt.title("Average Row Profile")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(row_var_profile)
plt.title("Row Variance Profile")
plt.xlabel("Row Index")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "row_artifact_analysis.png"))
plt.close()

# 3. Fixed Pattern Noise
plt.figure()
plt.imshow(fpn, cmap='seismic', vmin=-3*fpn_std, vmax=3*fpn_std)
plt.title(f"Fixed Pattern Noise (std={fpn_std:.2f})")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fpn_map.png"))
plt.close()

# 4. Sample FFT Analysis (using first image only)
img_fft = results['first_image'] - np.mean(results['first_image'])
f = fft2(img_fft)
fshift = fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure()
plt.imshow(magnitude_spectrum, cmap='inferno')
plt.title("FFT Magnitude Spectrum")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fft_spectrum.png"))
plt.close()

# ========================
# SAVE RESULTS
# ========================
print("[INFO] Saving summary metrics...")
summary_csv_path = os.path.join(output_folder, "summary_metrics.csv")
with open(summary_csv_path, mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Images Processed", num_images])
    writer.writerow(["Image Dimensions", f"{results['first_image'].shape[0]}x{results['first_image'].shape[1]}"])
    writer.writerow(["Global Mean", f"{global_mean:.2f}"])
    writer.writerow(["FPN Std", f"{fpn_std:.2f}"])
    writer.writerow(["Original Row Std", f"{original_row_std:.2f}"])
    writer.writerow(["Corrected Row Std", f"{corrected_row_std:.2f}"])
    writer.writerow(["Improvement Ratio", f"{improvement_ratio:.2f}x"])
    writer.writerow(["Mean CV", f"{np.mean(results['cvs']):.4f}"])
    writer.writerow(["Temporal Noise", f"{np.mean(results['stds']):.2f}"])

print(f"[DONE] Analysis complete. Results saved to:\n{output_folder}")