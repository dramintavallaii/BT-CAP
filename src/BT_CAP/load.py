import SimpleITK as sitk
import numpy as np
from typing import Tuple, Optional, Dict

def load_nifti(filepath: str, return_affine: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(img)
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32, copy=False)
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int16, copy=False)
    if not return_affine:
        return data, None
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return data, affine


def load_modalities_and_mask(paths_dict: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    t1n, affine = load_nifti(paths_dict["t1n_path"], return_affine=True)
    t1c, _ = load_nifti(paths_dict["t1c_path"])
    t2w, _ = load_nifti(paths_dict["t2w_path"])
    t2f, _ = load_nifti(paths_dict["t2f_path"])
    seg, _ = load_nifti(paths_dict["seg_path"])
    if paths_dict.get("mask_path") is not None:
        brain_mask, _ = load_nifti(paths_dict["mask_path"])
        brain_mask = brain_mask.astype(bool, copy=False)
    else:
        brain_mask = None
    modalities = np.stack([t1n, t1c, t2w, t2f], axis=0).astype(np.float32, copy=False)
    seg = seg.astype(np.uint8, copy=False)
    expected_shape = modalities.shape[1:]
    if seg.shape != expected_shape:
        raise ValueError(f"Segmentation shape {seg.shape} does not match modalities {expected_shape}.")
    if brain_mask is not None and brain_mask.shape != expected_shape:
        raise ValueError(f"Brain mask shape {brain_mask.shape} does not match modalities {expected_shape}.")
    return modalities, seg, brain_mask, affine