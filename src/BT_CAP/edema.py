from .utils import smooth_interface_with_kernel
from scipy.ndimage import zoom, binary_dilation, gaussian_filter, map_coordinates
import numpy as np

def apply_edema_deformation(modalities_, modalities__, mask__, original_tumor_core_, rng=None):
    edema_mask = (mask__ == 4).astype(np.uint8)
    tumor_core = original_tumor_core_ > 0
    if np.sum(edema_mask) == 0:
        return modalities__, edema_mask
    outer_shell = binary_dilation(edema_mask, iterations=5) & (~edema_mask)
    X, Y, Z = edema_mask.shape
    shape = np.array([X, Y, Z])
    grid_size = 16
    magnitude = rng.uniform(6.0, 12.0)
    lowres_shape = np.ceil(shape / grid_size).astype(int)
    rand_vectors = rng.normal(0, magnitude, size=(3, *lowres_shape))
    displacement_field = np.stack([
        gaussian_filter(
            zoom(rand_vectors[i], shape / np.array(lowres_shape), order=3),
            sigma=5
        )
        for i in range(3)
    ])
    displacement_field = displacement_field[:, :X, :Y, :Z]
    mask_to_freeze = tumor_core | (edema_mask & ~outer_shell)
    displacement_field[:, mask_to_freeze] = 0
    coords = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')
    coords = np.stack(coords, axis=0).astype(np.float32)
    warped_coords = coords + displacement_field
    warped_edema = map_coordinates(edema_mask.astype(np.float32), warped_coords, order=3, mode='constant')
    new_edema_mask = (warped_edema > 0.4).astype(np.uint8)
    deformed_modalities = np.zeros_like(modalities__, dtype=np.float32)
    for mod in range(modalities_.shape[0]):
        warped_modality = map_coordinates(modalities__[mod] * edema_mask, warped_coords, order=3, mode='constant')
        deformed_modalities[mod] = warped_modality.reshape(X, Y, Z)
    inner_edema = new_edema_mask & ~outer_shell
    for mod in range(deformed_modalities.shape[0]):
        deformed_modalities[mod] = smooth_interface_with_kernel(
            deformed_modalities[mod],
            new_edema_mask & outer_shell,
            inner_edema,
            dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3)
        )
    return deformed_modalities, new_edema_mask

def edema_management(volume, mask_, volume__, mask__, original_tumor_core_, labels_, rng=None):
    if 4 not in labels_:
        return volume, mask_
    deformed_edema_mods, deformed_edema_mask = apply_edema_deformation(
        volume, volume__, mask__, original_tumor_core_, rng=rng
    )
    output_mods = volume__.copy()
    output_mask = mask__.copy()
    output_mods[:, deformed_edema_mask > 0] = deformed_edema_mods[:, deformed_edema_mask > 0]
    output_mask[deformed_edema_mask > 0] = 4
    output_mods[:, mask_ > 0] = volume[:, mask_ > 0]
    output_mask[mask_ > 0] = mask_[mask_ > 0]
    for mod in range(output_mods.shape[0]):
        output_mods[mod] = smooth_interface_with_kernel(
            output_mods[mod], mask_ > 0, output_mask,
            dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3)
        )
    return output_mods, output_mask