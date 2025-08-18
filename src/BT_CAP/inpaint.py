from scipy.ndimage import gaussian_filter
import numpy as np
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure, gaussian_filter
from collections import Counter

def inpaint(modalities_, mask_, labels_, original_tumor_core_, threshold=10, sigma=1.0, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()
    output_modalities = modalities_.copy()
    output_mask = mask_.copy()
    empty_areas = (mask_ == 0) & original_tumor_core_.astype(bool)
    if not np.any(empty_areas):
        return output_modalities, output_mask
    struct = generate_binary_structure(3, 1)
    labeled, ncomps = label(empty_areas, structure=struct)
    for comp_id in range(1, ncomps + 1):
        island = (labeled == comp_id)
        coords = np.argwhere(island)
        size = coords.shape[0]
        donor_label = None
        if size <= threshold:
            border = binary_dilation(island, structure=struct) & (~island)
            neighbor_labels = mask_[border]
            neighbor_labels = neighbor_labels[neighbor_labels > 0]
            if neighbor_labels.size > 0:
                donor_label = Counter(neighbor_labels).most_common(1)[0][0]
        if donor_label is None:
            if 2 in labels_:
                donor_label = 2
            elif 3 in labels_:
                donor_label = 3
            else:
                donor_label = 1
        for i in range(4):
            donor_vals = modalities_[i, mask_ == donor_label]
            if donor_vals.size > 0:
                avg_intensity = np.mean(donor_vals)
                std_intensity = np.std(donor_vals)
            else:
                avg_intensity, std_intensity = 0.0, 0.0
            noisy_intensities = rng.normal(avg_intensity, std_intensity, size)
            temp_modality = modalities_[i].copy()
            temp_modality[island] = noisy_intensities
            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0) + 1
            pad = max(1, int(3 * sigma))
            z0, z1 = max(0, zmin - pad), min(temp_modality.shape[0], zmax + pad)
            y0, y1 = max(0, ymin - pad), min(temp_modality.shape[1], ymax + pad)
            x0, x1 = max(0, xmin - pad), min(temp_modality.shape[2], xmax + pad)
            local_cube = temp_modality[z0:z1, y0:y1, x0:x1]
            smoothed_cube = gaussian_filter(local_cube, sigma=sigma, mode='reflect')
            output_modalities[i][island] = smoothed_cube[island[z0:z1, y0:y1, x0:x1]]
        output_mask[island] = donor_label
    return output_modalities, output_mask