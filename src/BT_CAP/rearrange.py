from __future__ import annotations
from .utils import get_bounding_box, smooth_interface_with_kernel
from scipy.ndimage import label, center_of_mass, affine_transform, generate_binary_structure, gaussian_filter, binary_opening, convolve, distance_transform_edt
from scipy.optimize import minimize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.neighbors import NearestNeighbors
import numpy as np

def extract_sorted_islands(mask: np.ndarray) -> list[np.ndarray]:
    labeled, num_features = label(mask.astype(np.uint8))
    islands = []
    for i in range(1, num_features + 1):
        comp = (labeled == i)
        cnt = int(comp.sum())
        if cnt > 0:
            islands.append((cnt, comp))
    islands.sort(reverse=True, key=lambda x: x[0])
    return [comp for _, comp in islands]

def split_large_island(island: np.ndarray, threshold: int) -> list[tuple[int, np.ndarray]]:
    voxel_count = int(island.sum())
    if voxel_count <= threshold:
        return [(voxel_count, island.astype(np.uint8, copy=False))]
    dist = distance_transform_edt(island)
    if dist.max() < 1:
        return [(voxel_count, island.astype(np.uint8, copy=False))]
    coords = peak_local_max(dist, labels=island, num_peaks=2, exclude_border=False)
    if coords is None or len(coords) < 2:
        return [(voxel_count, island.astype(np.uint8, copy=False))]
    markers = np.zeros_like(island, dtype=np.int32)
    for i, c in enumerate(coords[:2], start=1):
        markers[tuple(c)] = i
    labels_ws = watershed(-dist, markers, mask=island)
    sub_islands = []
    for lbl in (1, 2):
        sub = (labels_ws == lbl)
        if sub.sum() >= 100:
            struct = generate_binary_structure(3, 1)
            sub = binary_opening(sub, structure=struct)
            cnt = int(sub.sum())
            if cnt > 0:
                sub_islands.append((cnt, sub.astype(np.uint8, copy=False)))
    if len(sub_islands) < 2:
        return [(voxel_count, island.astype(np.uint8, copy=False))]
    final = []
    for cnt, sub in sub_islands:
        final.extend(split_large_island(sub, threshold))
    return final

def transform_island(
    island: np.ndarray,
    angles_deg: np.ndarray,
    shifts_vox: np.ndarray,
    modalities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    theta_x, theta_y, theta_z = np.deg2rad(angles_deg.astype(float))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x),  np.cos(theta_x)]], dtype=np.float64)
    Ry = np.array([[ np.cos(theta_y), 0, np.sin(theta_y)],
                   [0,                1,            0   ],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype=np.float64)
    Rz = np.array([[ np.cos(theta_z), -np.sin(theta_z), 0],
                   [ np.sin(theta_z),  np.cos(theta_z), 0],
                   [0,                 0,                1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    center = (np.array(island.shape, dtype=np.float64) / 2.0)
    shifts = shifts_vox.astype(np.float64)
    offset = center - R @ center - shifts
    transformed_island = affine_transform(
        island.astype(np.float32, copy=False),
        R, offset=offset, order=0, mode="constant", cval=0.0, prefilter=False
    )
    transformed_island = (transformed_island > 0.5).astype(np.uint8)
    transformed_modalities = None
    if modalities is not None:
        C = modalities.shape[0]
        out = []
        for i in range(C):
            ch = affine_transform(
                modalities[i].astype(np.float32, copy=False),
                R, offset=offset, order=3, mode="constant", cval=0.0, prefilter=True
            )
            out.append(ch)
        transformed_modalities = np.stack(out, axis=0).astype(np.float32, copy=False)
        transformed_modalities *= transformed_island[None, ...]
    return transformed_island, transformed_modalities

def optimize_island(
    island: np.ndarray,
    tumor_core: np.ndarray,
    covered_core: np.ndarray,
    angle_bounds: tuple[float, float] = (-180.0, 180.0),
    shift_margin_vox: float | None = None,
    maxiter: int = 80,
    rng: np.random.Generator = None
) -> np.ndarray:
    empty_core = (tumor_core.astype(bool) & (~covered_core.astype(bool)))
    bbox = get_bounding_box(tumor_core, [1])
    if bbox is None:
        return np.zeros(6, dtype=np.float32)
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
    dz, dy, dx = (z_max - z_min), (y_max - y_min), (x_max - x_min)
    if shift_margin_vox is None:
        shift_margin_vox = 0.5
    shift_bounds = np.array([
        [-dx * shift_margin_vox,  dx * shift_margin_vox],
        [-dy * shift_margin_vox,  dy * shift_margin_vox],
        [-dz * shift_margin_vox,  dz * shift_margin_vox],
    ], dtype=np.float64)
    ang_lo, ang_hi = angle_bounds
    angle_bounds_arr = np.array([[ang_lo, ang_hi]] * 3, dtype=np.float64)
    com_island = np.array(center_of_mass(island))
    if empty_core.any():
        com_core = np.array(center_of_mass(empty_core))
    else:
        com_core = np.array(center_of_mass(tumor_core))
    init_shift = (com_core - com_island)
    init_shift = np.clip(init_shift, shift_bounds[:, 0], shift_bounds[:, 1])
    init_angles = rng.uniform(ang_lo / 4.0, ang_hi / 4.0, size=3)
    x0 = np.concatenate([init_angles, init_shift]).astype(np.float64)
    def _clamp_params(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.clip(p[:3], angle_bounds_arr[:, 0], angle_bounds_arr[:, 1])
        s = np.clip(p[3:], shift_bounds[:, 0], shift_bounds[:, 1])
        return a, s
    def cost(p: np.ndarray) -> float:
        a, s = _clamp_params(p)
        t_mask, _ = transform_island(island, a, s)
        sz = t_mask.sum()
        if sz == 0:
            return 1e6
        overlap = (t_mask.astype(bool) & empty_core).sum()
        score = overlap / float(sz)
        return -float(score)
    res = minimize(
        cost, x0, method="Powell",
        options={"maxiter": maxiter, "xtol": 0.2, "ftol": 0.1}
    )
    a, s = _clamp_params(res.x.astype(np.float64))
    return np.concatenate([a, s]).astype(np.float32)

def clean_mask_intensity_knn(volume: np.ndarray, mask: np.ndarray, mad_thresh=3.5, iqr_factor=1.5, k=5):
    mask = mask.astype(bool)
    tumor_voxels = volume[mask]
    if tumor_voxels.size == 0:
        return volume.copy(), np.zeros_like(volume, dtype=bool)
    tumor_coords = np.argwhere(mask)
    median_intensity = np.median(tumor_voxels)
    mad = np.median(np.abs(tumor_voxels - median_intensity))
    if mad > 0:
        modified_z = 0.6745 * (tumor_voxels - median_intensity) / mad
        out_idx = np.where(np.abs(modified_z) > mad_thresh)[0]
    else:
        q1, q3 = np.percentile(tumor_voxels, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        out_idx = np.where((tumor_voxels < lo) | (tumor_voxels > hi))[0]
    outlier_mask = np.zeros_like(volume, dtype=bool)
    if out_idx.size == 0:
        return volume.copy(), outlier_mask
    outlier_coords = tumor_coords[out_idx]
    outlier_mask[tuple(outlier_coords.T)] = True
    valid_coords = tumor_coords[np.setdiff1d(np.arange(len(tumor_coords)), out_idx)]
    valid_values = volume[tuple(valid_coords.T)]
    filled = volume.copy()
    if valid_coords.shape[0] == 0:
        return filled, outlier_mask
    nn = NearestNeighbors(n_neighbors=min(k, len(valid_coords)), algorithm="kd_tree")
    nn.fit(valid_coords)
    _, neigh_idx = nn.kneighbors(outlier_coords)
    for i, idxs in enumerate(neigh_idx):
        filled[tuple(outlier_coords[i])] = float(np.mean(valid_values[idxs]))
    return filled.astype(np.float32, copy=False), outlier_mask

def find_single_enclosed_empty_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    empty = ~mask
    struct = generate_binary_structure(3, 1).astype(np.uint8)
    neigh = convolve(mask.astype(np.uint8), struct, mode="constant", cval=0)
    enclosed = empty & (neigh == 6)
    return enclosed

def fill_enclosed_with_neighbor_mean(intensity_vol: np.ndarray, enclosed_mask: np.ndarray) -> np.ndarray:
    filled = intensity_vol.copy()
    Z, Y, X = intensity_vol.shape
    coords = np.argwhere(enclosed_mask)
    for z, y, x in coords:
        vals = []
        if z > 0:     vals.append(intensity_vol[z-1, y,   x  ])
        if z < Z-1:   vals.append(intensity_vol[z+1, y,   x  ])
        if y > 0:     vals.append(intensity_vol[z,   y-1, x  ])
        if y < Y-1:   vals.append(intensity_vol[z,   y+1, x  ])
        if x > 0:     vals.append(intensity_vol[z,   y,   x-1])
        if x < X-1:   vals.append(intensity_vol[z,   y,   x+1])
        if vals:
            filled[z, y, x] = float(np.mean(vals))
    return filled.astype(np.float32, copy=False)

def rearrange_subcomponents_into_core(
    tumor_core_dict: dict[int, tuple[np.ndarray, np.ndarray]],
    tumor_core: np.ndarray,
    case_name: str,
    sample_idx: int,
    queue=None,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    tumor_core = tumor_core.astype(np.uint8, copy=False)
    core_vox = int(tumor_core.sum())
    if core_vox == 0:
        return np.zeros_like(tumor_core, dtype=np.uint8), np.zeros((4, *tumor_core.shape), dtype=np.float32)
    threshold = max(100, core_vox // 2)
    enh_islands, nonenh_islands, cyst_islands = [], [], []
    for label_key, (sub_mask, sub_mods) in tumor_core_dict.items():
        for i_idx, island in enumerate(extract_sorted_islands(sub_mask)):
            if island.sum() < 100:
                continue
            for j_idx, (cnt, split_island) in enumerate(split_large_island(island, threshold)):
                if cnt < 100:
                    continue
                tup = (cnt,
                       f"Comp{label_key}-Island{i_idx}-Split{j_idx}",
                       split_island.astype(np.uint8, copy=False),
                       label_key,
                       sub_mods)
                if label_key == 1:
                    enh_islands.append(tup)
                elif label_key == 2:
                    nonenh_islands.append(tup)
                elif label_key == 3:
                    cyst_islands.append(tup)
    enh_islands.sort(key=lambda x: x[0], reverse=True)
    nonenh_islands.sort(key=lambda x: x[0], reverse=True)
    cyst_islands.sort(key=lambda x: x[0], reverse=True)
    total_islands = len(enh_islands) + len(nonenh_islands) + len(cyst_islands)
    if queue is not None:
        queue.put(("progress", case_name, sample_idx, 0, total_islands))
    covered_core = np.zeros_like(tumor_core, dtype=np.uint8)
    placed = 0
    placed_ETs_vol  = np.zeros((4, *tumor_core.shape), dtype=np.float32)
    placed_NETs_vol = np.zeros((4, *tumor_core.shape), dtype=np.float32)
    placed_CCs_vol  = np.zeros((4, *tumor_core.shape), dtype=np.float32)
    placed_ETs_mask  = np.zeros_like(tumor_core, dtype=np.uint8)
    placed_NETs_mask = np.zeros_like(tumor_core, dtype=np.uint8)
    placed_CCs_mask  = np.zeros_like(tumor_core, dtype=np.uint8)
    components = [
        (1, enh_islands,  "Enhancing"),
        (2, nonenh_islands, "Non-enhancing"),
        (3, cyst_islands, "Cystic"),
    ]
    while any(len(lst) > 0 for _, lst, _ in components):
        candidates = []
        for label_idx, lst, _ in components:
            if lst:
                cnt, island_id, island_mask, key_lbl, sub_mods = lst[0]
                candidates.append((cnt, island_mask, island_id, key_lbl, sub_mods, lst))
        candidates.sort(key=lambda t: t[0], reverse=True)
        for _, island_mask, island_id, key_lbl, sub_mods, source_list in candidates:
            params = optimize_island(island_mask, tumor_core, covered_core, rng=rng)
            t_mask, t_mods = transform_island(island_mask, params[:3], params[3:], modalities=sub_mods)
            placed += 1
            if queue is not None:
                queue.put(("island_placement", case_name, sample_idx, placed, total_islands))

            if t_mask.any():
                t_mask = gaussian_filter(t_mask.astype(np.float32), sigma=1.0) > 0.6
                t_mask = t_mask.astype(np.uint8, copy=False)
                if t_mods is not None:
                    t_mods *= t_mask[None, ...]
            within_core = (t_mask > 0) & (tumor_core > 0)
            if within_core.any() and t_mods is not None:
                if key_lbl == 1:
                    placed_ETs_vol[:, within_core] = t_mods[:, within_core]
                    for m in range(4):
                        placed_ETs_vol[m] = smooth_interface_with_kernel(
                            placed_ETs_vol[m], t_mask.astype(bool), placed_ETs_mask.astype(bool),
                            dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3)
                        )
                    placed_ETs_mask[within_core] = 1
                elif key_lbl == 2:
                    placed_NETs_vol[:, within_core] = t_mods[:, within_core]
                    for m in range(4):
                        placed_NETs_vol[m] = smooth_interface_with_kernel(
                            placed_NETs_vol[m], t_mask.astype(bool), placed_NETs_mask.astype(bool),
                            dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3)
                        )
                    placed_NETs_mask[within_core] = 1
                elif key_lbl == 3:
                    placed_CCs_vol[:, within_core] = t_mods[:, within_core]
                    for m in range(4):
                        placed_CCs_vol[m] = smooth_interface_with_kernel(
                            placed_CCs_vol[m], t_mask.astype(bool), placed_CCs_mask.astype(bool),
                            dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3)
                        )
                    placed_CCs_mask[within_core] = 1
                covered_core[within_core] = 1
            source_list.pop(0)
    ET_holes  = find_single_enclosed_empty_voxels(placed_ETs_mask)
    NET_holes = find_single_enclosed_empty_voxels(placed_NETs_mask)
    CC_holes  = find_single_enclosed_empty_voxels(placed_CCs_mask)
    cleaned_ETs  = np.zeros_like(placed_ETs_vol,  dtype=np.float32)
    cleaned_NETs = np.zeros_like(placed_NETs_vol, dtype=np.float32)
    cleaned_CCs  = np.zeros_like(placed_CCs_vol,  dtype=np.float32)
    for m in range(4):
        if placed_ETs_mask.any():
            cleaned_ETs[m], _ = clean_mask_intensity_knn(placed_ETs_vol[m], placed_ETs_mask)
            cleaned_ETs[m] = fill_enclosed_with_neighbor_mean(cleaned_ETs[m], ET_holes)
            placed_ETs_mask[ET_holes] = 1
        if placed_NETs_mask.any():
            cleaned_NETs[m], _ = clean_mask_intensity_knn(placed_NETs_vol[m], placed_NETs_mask)
            cleaned_NETs[m] = fill_enclosed_with_neighbor_mean(cleaned_NETs[m], NET_holes)
            placed_NETs_mask[NET_holes] = 1
        if placed_CCs_mask.any():
            cleaned_CCs[m], _ = clean_mask_intensity_knn(placed_CCs_vol[m], placed_CCs_mask)
            cleaned_CCs[m] = fill_enclosed_with_neighbor_mean(cleaned_CCs[m], CC_holes)
            placed_CCs_mask[CC_holes] = 1
    final_vol  = np.zeros_like(placed_ETs_vol, dtype=np.float32)
    final_mask = np.zeros_like(tumor_core,    dtype=np.uint8)
    inter_mask = np.zeros_like(tumor_core,    dtype=np.uint8)
    if placed_NETs_mask.any():
        final_vol[:, placed_NETs_mask > 0] = cleaned_NETs[:, placed_NETs_mask > 0]
        for m in range(4):
            final_vol[m] = smooth_interface_with_kernel(final_vol[m], placed_NETs_mask.astype(bool), inter_mask.astype(bool),
                                                        dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3))
        inter_mask[placed_NETs_mask > 0] = 1
        final_mask[placed_NETs_mask > 0] = 2
    if placed_CCs_mask.any():
        final_vol[:, placed_CCs_mask > 0] = cleaned_CCs[:, placed_CCs_mask > 0]
        for m in range(4):
            final_vol[m] = smooth_interface_with_kernel(final_vol[m], placed_CCs_mask.astype(bool), inter_mask.astype(bool),
                                                        dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3))
        inter_mask[placed_CCs_mask > 0] = 1
        final_mask[placed_CCs_mask > 0] = 3
    if placed_ETs_mask.any():
        final_vol[:, placed_ETs_mask > 0] = cleaned_ETs[:, placed_ETs_mask > 0]
        for m in range(4):
            final_vol[m] = smooth_interface_with_kernel(final_vol[m], placed_ETs_mask.astype(bool), inter_mask.astype(bool),
                                                        dilate_iter=2, alpha=0.7, kernel_size=(3, 3, 3))
        inter_mask[placed_ETs_mask > 0] = 1
        final_mask[placed_ETs_mask > 0] = 1
    return final_mask, final_vol