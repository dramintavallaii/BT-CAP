from .utils import np_to_sitk, sitk_to_np
from scipy.ndimage import generate_binary_structure, binary_opening
import SimpleITK as sitk
import numpy as np

def extract_bounding_box_crop(mods, mask, affine, original_shape, margin: int = 10):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return mods, mask, affine, (0, original_shape[0], 0, original_shape[1], 0, original_shape[2])
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    z1, z2 = max(0, z_min - margin), min(z_max + margin, mask.shape[0])
    y1, y2 = max(0, y_min - margin), min(y_max + margin, mask.shape[1])
    x1, x2 = max(0, x_min - margin), min(x_max + margin, mask.shape[2])
    cropped_mask = mask[z1:z2, y1:y2, x1:x2]
    cropped_mods = mods[:, z1:z2, y1:y2, x1:x2]
    offset = np.array([z1, y1, x1], dtype=float)
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ offset
    bbox = (z1, z2, y1, y2, x1, x2)
    return cropped_mods, cropped_mask, new_affine, bbox

def compute_adaptive_params(mask, affine, base_spacing=10.0, base_displacement=5.0):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (base_spacing,) * 3, base_displacement
    bbox = coords.max(axis=0) - coords.min(axis=0) + 1
    spacing = [np.linalg.norm(affine[:3, i]) for i in range(3)]
    physical_bbox = [bbox[i] * spacing[i] for i in range(3)]
    mask_volume_vox = coords.shape[0]
    voxel_volume_mm3 = np.abs(np.linalg.det(affine[:3, :3]))
    volume_mm3 = mask_volume_vox * voxel_volume_mm3
    scale_factor = (volume_mm3 / 10000.0) ** (1 / 3)
    scale_factor = max(0.5, min(2.0, scale_factor))
    grid_spacing = tuple(base_spacing * scale_factor for _ in range(3))
    min_phys_dim = min(physical_bbox)
    displacement = base_displacement * max(0.5, min(2.0, min_phys_dim / 50.0))
    return grid_spacing, displacement

def apply_bspline_deformation(
    volume_np,
    mask_np=None,
    affine=None,
    grid_physical_spacing=None,
    bspline_order=3,
    random_displacement=None,
    disp_scale=3.0,
    rng: np.random.Generator = None
):
    if rng is None:
        rng = np.random.default_rng()
    if grid_physical_spacing is None or random_displacement is None:
        grid_physical_spacing, random_displacement = compute_adaptive_params(mask_np, affine)
    sitk_vol = np_to_sitk(volume_np, affine)
    sitk_mask = None
    if mask_np is not None:
        sitk_mask = sitk.GetImageFromArray(mask_np.astype(np.uint8))
        sitk_mask.CopyInformation(sitk_vol)
    size = sitk_vol.GetSize()
    spacing = sitk_vol.GetSpacing()
    grid_size = [int(np.ceil(sz * spc / gps)) for sz, spc, gps in zip(size, spacing, grid_physical_spacing)]
    transform = sitk.BSplineTransformInitializer(sitk_vol, grid_size, bspline_order)
    params = np.array(transform.GetParameters(), dtype=float)
    displacement = (rng.random(params.size) * 2 * disp_scale - disp_scale) * random_displacement
    transform.SetParameters(tuple(params + displacement))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_vol)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    deformed_vol = resampler.Execute(sitk_vol)
    deformed_vol_np = sitk_to_np(deformed_vol)
    deformed_mask_np = None
    if sitk_mask is not None:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        deformed_mask = resampler.Execute(sitk_mask)
        deformed_mask_np = sitk.GetArrayFromImage(deformed_mask).astype(np.uint8)
        if deformed_mask_np.sum() > 0:
            struct = generate_binary_structure(3, 1)
            smoothed_mask = binary_opening(deformed_mask_np, structure=struct)
            deformed_vol_np *= smoothed_mask[np.newaxis, ...]
            deformed_mask_np = smoothed_mask
    return deformed_vol_np, deformed_mask_np