import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation, uniform_filter
from typing import List, Optional, Tuple

def np_to_sitk(img_np, affine):
    if img_np.ndim == 4:
        sitk_channels = [sitk.GetImageFromArray(img_np[c]) for c in range(img_np.shape[0])]
        sitk_img = sitk.Compose(sitk_channels)
    else:
        sitk_img = sitk.GetImageFromArray(img_np)
    spacing = np.linalg.norm(affine[:3, :3], axis=0).astype(float)
    direction_matrix = affine[:3, :3] / spacing
    u, _, vh = np.linalg.svd(direction_matrix)
    direction = (u @ vh).flatten().astype(float)
    origin = affine[:3, 3].astype(float)
    sitk_img.SetSpacing([float(s) for s in spacing])
    sitk_img.SetDirection([float(d) for d in direction])
    sitk_img.SetOrigin([float(o) for o in origin])
    return sitk_img

def sitk_to_np(img_sitk: sitk.Image) -> np.ndarray:
    comps = img_sitk.GetNumberOfComponentsPerPixel()
    if comps > 1:
        return np.asarray([
            sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(img_sitk, i))
            for i in range(comps)
        ])
    else:
        return sitk.GetArrayFromImage(img_sitk)
    
def get_islands(mask: np.ndarray) -> list[np.ndarray]:
    labeled_mask, num_features = label(mask)
    return [(labeled_mask == i) for i in range(1, num_features + 1)]

def get_bounding_box(
    mask: np.ndarray, 
    label_values: int | List[int]
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    if np.isscalar(label_values):
        label_values = [int(label_values)]
    else:
        label_values = [int(l) for l in label_values]
    indices = np.where(np.isin(mask, label_values))
    if indices[0].size == 0:
        return None
    z_min, y_min, x_min = [int(np.min(idx)) for idx in indices]
    z_max, y_max, x_max = [int(np.max(idx)) + 1 for idx in indices]
    return (z_min, z_max), (y_min, y_max), (x_min, x_max)

def ZScoreNormalize(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if not np.any(mask):
        mean = volume.mean()
        std = volume.std() + 1e-8
    else:
        mean = volume[mask].mean()
        std = volume[mask].std() + 1e-8
    return (volume - mean) / std

def smooth_interface_with_kernel(
    volume: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    dilate_iter: int = 2,
    alpha: float = 0.7,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
) -> np.ndarray:
    m1 = mask1.astype(bool, copy=False)
    m2 = mask2.astype(bool, copy=False)
    interface = m1 & binary_dilation(m2, iterations=dilate_iter)
    if np.sum(interface) > 0:
        smoothed = volume.copy()
        dz, dy, dx = kernel_size
        pad_z, pad_y, pad_x = dz // 2, dy // 2, dx // 2
        padded_volume = np.pad(volume, 
                                ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 
                                mode='reflect')
        foreground_mask = (mask1 | mask2).astype(np.uint8)
        padded_mask = np.pad(foreground_mask, 
                             ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 
                             mode='constant', constant_values=0)
        interface_idx = np.argwhere(interface)
        for z, y, x in interface_idx:
            patch = padded_volume[z : z + dz, y : y + dy, x : x + dx]
            patch_mask = padded_mask[z : z + dz, y : y + dy, x : x + dx]
            if np.any(patch_mask):
                patch_mean = np.mean(patch[patch_mask > 0])
                smoothed[z, y, x] = (1 - alpha) * volume[z, y, x] + alpha * patch_mean
        return smoothed
    return volume

# def smooth_interface_with_kernel(
#     volume: np.ndarray,
#     mask1: np.ndarray,
#     mask2: np.ndarray,
#     dilate_iter: int = 2,
#     alpha: float = 0.7,
#     kernel_size: tuple[int, int, int] = (3, 3, 3),
# ) -> np.ndarray:
#     m1 = mask1.astype(bool, copy=False)
#     m2 = mask2.astype(bool, copy=False)
#     interface = m1 & binary_dilation(m2, iterations=dilate_iter)
#     if not np.any(interface):
#         return volume
#     pad = tuple(k // 2 for k in kernel_size)
#     idx = np.where(interface)
#     z0, z1 = max(int(idx[0].min()) - pad[0] - 1, 0), min(int(idx[0].max()) + pad[0] + 2, volume.shape[0])
#     y0, y1 = max(int(idx[1].min()) - pad[1] - 1, 0), min(int(idx[1].max()) + pad[1] + 2, volume.shape[1])
#     x0, x1 = max(int(idx[2].min()) - pad[2] - 1, 0), min(int(idx[2].max()) + pad[2] + 2, volume.shape[2])
#     v_crop = volume[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
#     i_crop = interface[z0:z1, y0:y1, x0:x1]
#     f_crop = (m1 | m2)[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
#     vol_mean = uniform_filter(v_crop, size=kernel_size, mode="reflect")
#     occ_mean = uniform_filter(f_crop, size=kernel_size, mode="constant")
#     with np.errstate(invalid="ignore", divide="ignore"):
#         local_mean = np.divide(vol_mean, occ_mean, where=occ_mean > 0)
#     out = v_crop.copy()
#     sel = i_crop & (occ_mean > 0)
#     out[sel] = (1.0 - alpha) * v_crop[sel] + alpha * local_mean[sel]
#     volume[z0:z1, y0:y1, x0:x1] = out
#     return volume

def insert_back_to_original_shape(
    resized_vol: np.ndarray,
    original_bbox: tuple,
    original_shape: tuple[int, int, int],
) -> np.ndarray:
    z1, z2, y1, y2, x1, x2 = original_bbox
    orig_center = [(z1 + z2) // 2, (y1 + y2) // 2, (x1 + x2) // 2]
    if resized_vol.ndim == 4:
        c, d, h, w = resized_vol.shape
    else:
        d, h, w = resized_vol.shape
    def get_start_end(center, size, max_len):
        start = center - size // 2
        end = start + size
        paste_start = max(start, 0)
        paste_end = min(end, max_len)
        extract_start = max(0, paste_start - start)
        extract_end = extract_start + (paste_end - paste_start)
        return paste_start, paste_end, extract_start, extract_end
    z_start, z_end, rz_start, rz_end = get_start_end(orig_center[0], d, original_shape[0])
    y_start, y_end, ry_start, ry_end = get_start_end(orig_center[1], h, original_shape[1])
    x_start, x_end, rx_start, rx_end = get_start_end(orig_center[2], w, original_shape[2])
    if resized_vol.ndim == 4:
        result = np.zeros((resized_vol.shape[0], *original_shape), dtype=resized_vol.dtype)
        result[:, z_start:z_end, y_start:y_end, x_start:x_end] = resized_vol[:, rz_start:rz_end, ry_start:ry_end, rx_start:rx_end]
    else:
        result = np.zeros(original_shape, dtype=resized_vol.dtype)
        result[z_start:z_end, y_start:y_end, x_start:x_end] = resized_vol[rz_start:rz_end, ry_start:ry_end, rx_start:rx_end]
    return result