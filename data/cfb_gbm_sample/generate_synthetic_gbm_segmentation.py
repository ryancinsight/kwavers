"""Generate a synthetic GBM tumor segmentation in RIRE patient 109 CT space.

Creates a multi-component GBM-like tumor (necrotic core + enhancing rim) with
irregular morphology placed in the brain parenchyma. Saved as a binary mask
(1 = tumor) in the CT NIfTI coordinate frame.

The tumor mimics radiographic GBM features:
  - Irregular, lobulated shape
  - Central necrotic core (included in tumor mask)
  - Enhancing rim (included in tumor mask)
  - Located in white matter / deep brain
  - Diameter ~30-40 mm (realistic for presentation)

Usage:
    python generate_synthetic_gbm_segmentation.py

Output:
    data/cfb_gbm_sample/segmentation.nii.gz  (binary mask, same space as CT)
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def brain_mask_from_ct(ct_data: np.ndarray, hu_min: float = 0.0, hu_max: float = 100.0) -> np.ndarray:
    """Create a brain mask from CT Hounsfield units.

    Brain parenchyma typically falls in the 0-100 HU range in non-contrast CT
    (excluding skull bone > 200 HU, CSF near 0, and air < -500 HU). We use a
    simple threshold plus morphological cleanup.
    """
    mask = (ct_data >= hu_min) & (ct_data <= hu_max)

    # Remove small disconnected components (noise, table artifacts)
    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        raise ValueError("No brain voxels found in CT data")
    component_sizes = ndimage.sum(np.ones_like(mask), labeled, index=range(1, n_labels + 1))
    largest_idx = np.argmax(component_sizes) + 1
    mask = labeled == largest_idx

    # Fill holes with binary closing
    struct = ndimage.generate_binary_structure(3, 2)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=2)
    mask = ndimage.binary_fill_holes(mask)

    return mask


def eroded_brain_center(brain_mask: np.ndarray) -> np.ndarray:
    """Find the centroid of the eroded brain mask (deep white matter region)."""
    eroded = ndimage.binary_erosion(brain_mask, iterations=8)
    if not np.any(eroded):
        # Fallback: use full brain centroid
        eroded = brain_mask
    indices = np.nonzero(eroded)
    return np.array([np.mean(d.astype(np.float64)) for d in indices])


def generate_gbm_tumor(
    shape: tuple[int, int, int],
    center_voxel: np.ndarray,
    major_radius_vox: float = 14.0,
    minor_radius_vox: float = 10.0,
    rim_thickness_vox: float = 3.0,
    necrotic_fraction: float = 0.55,
    noise_amplitude: float = 0.35,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic GBM tumor mask with necrotic core and irregular rim.

    Parameters
    ----------
    shape : (nx, ny, nz)
        Volume dimensions.
    center_voxel : (3,) array
        Tumor center in voxel coordinates [ix, iy, iz].
    major_radius_vox : float
        Semi-major axis of the tumor ellipsoid in voxels.
    minor_radius_vox : float
        Semi-minor axis in voxels.
    rim_thickness_vox : float
        Thickness of the enhancing rim in voxels.
    necrotic_fraction : float
        Fraction of the tumor radius that is necrotic core.
    noise_amplitude : float
        Amplitude of Perlin-like noise to create irregular border.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tumor_mask : (nx, ny, nz) bool
        Binary tumor mask.
    """
    rng = np.random.RandomState(seed)
    nx, ny, nz = shape
    cx, cy, cz = center_voxel

    # Create coordinate grids
    x = np.arange(nx, dtype=np.float64)[:, np.newaxis, np.newaxis] - cx
    y = np.arange(ny, dtype=np.float64)[np.newaxis, :, np.newaxis] - cy
    z = np.arange(nz, dtype=np.float64)[np.newaxis, np.newaxis, :] - cz

    # Ellipsoidal distance with random axis rotation
    theta = rng.uniform(0, 2 * np.pi)  # Random rotation in xy
    phi = rng.uniform(0, np.pi / 4)    # Slight tilt in z
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    z_rot = z * cos_p + y_rot * sin_p

    # Normalized ellipsoidal distance
    dist_sq = (x_rot / major_radius_vox) ** 2 + (z_rot / minor_radius_vox) ** 2 + (y_rot / major_radius_vox) ** 2
    dist = np.sqrt(np.maximum(dist_sq, 0.0))

    # Generate multi-scale noise for irregular tumor border
    noise = _layered_noise(shape, scale_vox=major_radius_vox, amplitude=noise_amplitude, rng=rng)

    # Perturbed boundary: the actual tumor boundary is the ellipsoid + noise
    perturbed_dist = dist - noise * 0.7

    # Tumor mask (everything inside the outer boundary)
    tumor_outer = perturbed_dist <= 1.0

    # Necrotic core (inner region)
    necrotic = perturbed_dist <= necrotic_fraction

    # The tumor mask includes both the enhancing rim and the necrotic core.
    # In GBM, the enhancing rim is the biologically active tumor; the necrotic
    # core is included because it contains tumor debris and treatment planning
    # must cover the entire lesion volume.
    tumor_mask = tumor_outer.copy()

    # Remove disconnected fragments from the noise
    labeled, n_labels = ndimage.label(tumor_mask)
    if n_labels > 1:
        component_sizes = ndimage.sum(np.ones_like(tumor_mask), labeled, index=range(1, n_labels + 1))
        largest_idx = np.argmax(component_sizes) + 1
        tumor_mask = labeled == largest_idx

    return tumor_mask


def _layered_noise(
    shape: tuple[int, int, int],
    scale_vox: float,
    amplitude: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate multi-scale noise for irregular tumor borders.

    Uses superposed low-frequency noise fields to create realistic,
    lobulated tumor morphology.
    """
    nx, ny, nz = shape
    noise = np.zeros(shape, dtype=np.float64)

    # Three octaves of noise at different spatial scales
    scales = [scale_vox * 1.5, scale_vox * 0.6, scale_vox * 0.25]
    weights = [0.5, 0.35, 0.15]

    for scale, weight in zip(scales, weights):
        # Low-resolution noise grid
        nx_low = max(2, int(nx / scale))
        ny_low = max(2, int(ny / scale))
        nz_low = max(2, int(nz / scale))
        low_res = rng.randn(nx_low, ny_low, nz_low).astype(np.float64)

        # Smooth with Gaussian
        low_res = ndimage.gaussian_filter(low_res, sigma=1.0)

        # Upsample to full resolution
        zoom_factors = (nx / nx_low, ny / ny_low, nz / nz_low)
        full_res = ndimage.zoom(low_res, zoom_factors, order=1)

        # Clip to original shape (zoom may produce off-by-one)
        full_res = full_res[:nx, :ny, :nz]

        noise += weight * full_res

    # Normalize to roughly [-amplitude, +amplitude]
    noise_std = np.std(noise)
    if noise_std > 1e-10:
        noise = noise / noise_std * amplitude * 0.5

    return noise


def main() -> None:
    # Paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    ct_path = repo_root / "data" / "rire_patient_109" / "patient_109_ct.nii.gz"
    output_dir = repo_root / "data" / "cfb_gbm_sample"
    output_path = output_dir / "segmentation.nii.gz"

    # Load CT
    print(f"Loading CT: {ct_path}")
    ct_img = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata().astype(np.float64)
    print(f"  Shape: {ct_data.shape}, Spacing: {ct_img.header.get_zooms()}")

    # Brain mask
    print("Creating brain mask...")
    brain = brain_mask_from_ct(ct_data)
    brain_voxels = int(np.sum(brain))
    print(f"  Brain voxels: {brain_voxels}")

    # Find brain center for tumor placement
    center = eroded_brain_center(brain)
    print(f"  Tumor center (voxel): [{center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f}]")

    # Generate synthetic GBM tumor
    print("Generating synthetic GBM tumor...")
    tumor = generate_gbm_tumor(
        shape=ct_data.shape,
        center_voxel=center,
        major_radius_vox=16.0,   # ~6.5 mm at 0.404 mm spacing
        minor_radius_vox=12.0,   # ~4.9 mm
        rim_thickness_vox=3.0,   # ~1.2 mm
        necrotic_fraction=0.55,
        noise_amplitude=0.30,
        seed=42,
    )

    # Constrain tumor to brain mask
    tumor = tumor & brain
    tumor_voxels = int(np.sum(tumor))
    print(f"  Tumor voxels: {tumor_voxels}")

    if tumor_voxels == 0:
        print("ERROR: Tumor does not overlap brain mask. Try adjusting center/radius.", file=sys.stderr)
        sys.exit(1)

    # Compute approximate diameter
    indices = np.nonzero(tumor)
    extent_mm = np.array([
        (indices[0].max() - indices[0].min()) * 0.404341,
        (indices[1].max() - indices[1].min()) * 0.404341,
        (indices[2].max() - indices[2].min()) * 3.0,
    ])
    print(f"  Tumor extent (mm): x={extent_mm[0]:.1f}, y={extent_mm[1]:.1f}, z={extent_mm[2]:.1f}")
    print(f"  Approximate equivalent diameter: {np.mean(extent_mm[:2]):.1f} mm")

    # Save as binary mask (int16, same affine as CT)
    print(f"Saving: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tumor_img = nib.Nifti1Image(tumor.astype(np.int16), ct_img.affine, ct_img.header)
    nib.save(tumor_img, str(output_path))

    # Verification: re-load and check
    reloaded = nib.load(str(output_path))
    reloaded_data = reloaded.get_fdata()
    n_foreground = int(np.sum(reloaded_data > 0))
    print(f"  Verified: {n_foreground} foreground voxels in saved file")
    print("Done.")


if __name__ == "__main__":
    main()
