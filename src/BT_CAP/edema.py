from .utils import smooth_interface_with_kernel
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
import numpy as np

def apply_edema_deformation(modalities_, modalities__, mask__, original_tumor_core_, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()
    edema_mask = (mask__ == 4).astype(np.uint8)
    new_edema_mask = edema_mask.copy()
    outer_shell = binary_dilation(edema_mask, iterations=5) & (~edema_mask)
    tumor_core = original_tumor_core_ > 0
    X, Y, Z = edema_mask.shape
    shape = (X, Y, Z)
    grid_size = 16
    magnitude = 8.0
    lowres_shape = ((np.array(shape) + grid_size - 1) // grid_size).astype(int)
    rand_vectors = rng.normal(loc=0.0, scale=magnitude, size=(3, *lowres_shape))
    displacement_field = np.stack([
        gaussian_filter(np.kron(rand_vectors[i], np.ones((grid_size, grid_size, grid_size))), sigma=5)
        for i in range(3)
    ])
    displacement_field = displacement_field[:, :X, :Y, :Z]
    mask_to_freeze = tumor_core | (edema_mask & ~outer_shell)
    freeze_mask_4d = mask_to_freeze[np.newaxis, :, :, :]
    displacement_field = np.where(freeze_mask_4d, 0, displacement_field)
    x, y, z = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')
    coords = np.stack([x, y, z], axis=0).astype(np.float32)
    warped_coords = coords + displacement_field
    new_edema_mask = map_coordinates(edema_mask.astype(np.float32), warped_coords, order=3, mode='constant', cval=0.0)
    new_edema_mask = (new_edema_mask > 0.4).astype(np.uint8)
    deformed_modalities = np.zeros_like(modalities__)
    for mod in range(modalities_.shape[0]):
        deformed_modality_flat = map_coordinates(
            modalities__[mod] * edema_mask, warped_coords, order=3, mode='constant', cval=0.0
        )
        deformed_modalities[mod] = deformed_modality_flat.reshape(X, Y, Z)
    inner_edema = new_edema_mask & ~outer_shell
    for mod in range(deformed_modalities.shape[0]):
        smoothed_modality = smooth_interface_with_kernel(
            deformed_modalities[mod], new_edema_mask & outer_shell, inner_edema, dilate_iter=2,
            alpha=0.7, kernel_size=(3, 3, 3)
        )
        deformed_modalities[mod] = smoothed_modality
    return deformed_modalities, new_edema_mask

def edema_management(volume, mask_, volume__, mask__, original_tumor_core_, labels_, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()
    output_mask = np.zeros_like(mask__).astype(np.uint8)
    output_mods = np.zeros_like(volume__).astype(np.float32)
    if 4 in labels_:
        deformed_edema_mods, deformed_edema_mask = apply_edema_deformation(volume, volume__, mask__, original_tumor_core_, rng=rng)
        output_mask[deformed_edema_mask > 0] = 4
        output_mods[:, deformed_edema_mask > 0] = deformed_edema_mods[:, deformed_edema_mask > 0]
        output_mods[:, mask_ > 0] = volume[:, mask_ > 0]
        for mod in range(output_mods.shape[0]):
            output_mods[mod] = smooth_interface_with_kernel(output_mods[mod], mask_ > 0, output_mask, dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3))
        output_mask[mask_ > 0] = mask_[mask_ > 0]
    else:
        output_mask = mask_
        output_mods = volume
    return output_mods, output_mask