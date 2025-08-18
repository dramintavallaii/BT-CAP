from .utils import get_bounding_box
import numpy as np
from scipy.ndimage import generate_binary_structure, binary_opening

def crop_volume(volume: np.ndarray, bbox: tuple) -> np.ndarray:
    if bbox is None:
        return np.zeros((0,), dtype=volume.dtype)
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
    if volume.ndim == 4:
        return volume[:, z_min:z_max, y_min:y_max, x_min:x_max]
    return volume[z_min:z_max, y_min:y_max, x_min:x_max]

def extract_target_component_only(
    modalities: np.ndarray,
    mask: np.ndarray,
    label_value: int,
    min_voxels: int = 100,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    bbox = get_bounding_box(mask, label_value)
    if bbox is None:
        return np.zeros((0,), dtype=modalities.dtype), np.zeros((0,), dtype=np.uint8), []
    cropped_modalities = crop_volume(modalities, bbox)
    cropped_mask = crop_volume(mask, bbox)
    binary_mask = (cropped_mask == label_value).astype(np.uint8)
    if binary_mask.sum() < min_voxels:
        return cropped_modalities * binary_mask[None], binary_mask, list(np.array(bbox).flatten())
    if smooth:
        structure = generate_binary_structure(3, 1)
        smoothed = binary_opening(binary_mask, structure=structure)
        if smoothed.sum() > 0:
            binary_mask = smoothed.astype(np.uint8, copy=False)
    if cropped_modalities.ndim == 4:
        cropped_modalities = cropped_modalities * binary_mask[None]
    else:
        cropped_modalities = cropped_modalities * binary_mask
    return cropped_modalities, binary_mask, list(np.array(bbox).flatten())