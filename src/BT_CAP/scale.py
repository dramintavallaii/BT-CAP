import numpy as np
from scipy.ndimage import zoom

def random_rescale(
    volume: np.ndarray,
    mask: np.ndarray,
    scale: float,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    assert volume.shape[0] == 4, "Expected (4, Z, Y, X) input"
    assert volume.shape[1:] == mask.shape, "Volume/mask shape mismatch"
    zoom_factors = (scale, scale, scale)
    if mask.sum() > 0:
        ranges = [(np.min(volume[i][mask > 0]), np.max(volume[i][mask > 0])) for i in range(4)]
    else:
        ranges = [(volume[i].min(), volume[i].max()) for i in range(4)]
    resized_modalities = []
    for i in range(4):
        resized = zoom(volume[i].astype(np.float32), zoom=zoom_factors, order=order, mode="constant")
        lo, hi = ranges[i]
        if hi > lo:
            resized = np.clip(resized, lo, hi)
        resized_modalities.append(resized)
    resized_modalities = np.stack(resized_modalities, axis=0)
    resized_mask = zoom(mask, zoom=zoom_factors, order=0, mode="nearest").astype(np.uint8)
    resized_modalities *= resized_mask[None]
    return resized_modalities, resized_mask
