import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import SimpleITK as sitk
from .utils import np_to_sitk, ZScoreNormalize, smooth_interface_with_kernel
from .load import load_modalities_and_mask
from .extract import extract_target_component_only
from .scale import random_rescale
from .deform import extract_bounding_box_crop, apply_bspline_deformation
from .rearrange import rearrange_subcomponents_into_core
from .inpaint import inpaint
from .edema import edema_management

REQUIRED_KEYS = ("t1n_path", "t1c_path", "t2w_path", "t2f_path", "seg_path")

def _discover_case(paths_dir: Path) -> Dict[str, Optional[str]]:
    case_dict: Dict[str, Optional[str]] = {
        "t1n_path": None,
        "t1c_path": None,
        "t2w_path": None,
        "t2f_path": None,
        "seg_path": None,
        "mask_path": None,
    }
    for f in sorted(paths_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name.lower()
        if name.startswith("."):
            continue
        if not any(name.endswith(ext) for ext in (".nii", ".nii.gz")):
            continue
        p = str(f)
        if "t1n" in name:
            case_dict["t1n_path"] = p
        elif "t1c" in name:
            case_dict["t1c_path"] = p
        elif "t2w" in name:
            case_dict["t2w_path"] = p
        elif "t2f" in name:
            case_dict["t2f_path"] = p
        elif "seg" in name:
            case_dict["seg_path"] = p
        elif "mask" in name:
            case_dict["mask_path"] = p
    missing = [k for k in REQUIRED_KEYS if case_dict[k] is None]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {paths_dir}: {missing}. "
            f"Found: {case_dict}"
        )
    return case_dict

def process_case(paths_dict: Dict[str, str], rng: Optional[np.random.Generator] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng()
    modalities, seg, brain_mask, affine = load_modalities_and_mask(paths_dict)
    modalities = modalities.astype(np.float32, copy=False)
    seg = seg.astype(np.uint8, copy=False)
    if brain_mask is not None:
        brain_mask = brain_mask.astype(bool, copy=False)
    labels = np.unique(seg)[1:]
    original_tumor_core = np.logical_or.reduce([(seg == 1), (seg == 2), (seg == 3)]).astype(np.uint8)
    original_shape = seg.shape
    tcs_segs_mods: Dict[int, list] = {}
    for label_ in labels:
        if label_ == 4:
            continue
        extracted_mods, extracted_mask, extracted_bbox = extract_target_component_only(
            modalities, seg, label_value=label_
        )
        scale = rng.uniform(1.3, 1.5)
        rescaled_mods, rescaled_mask = random_rescale(extracted_mods, extracted_mask, scale=scale)
        from .utils import insert_back_to_original_shape
        reinserted_mods = insert_back_to_original_shape(rescaled_mods, extracted_bbox, original_shape)
        reinserted_mask = insert_back_to_original_shape(rescaled_mask, extracted_bbox, original_shape)
        cropped_mods, cropped_mask, cropped_affine, cropped_bbox = extract_bounding_box_crop(
            reinserted_mods, reinserted_mask, affine, original_shape=original_shape
        )
        deformed_mods, deformed_mask = apply_bspline_deformation(
            cropped_mods, cropped_mask, cropped_affine,
            grid_physical_spacing=None, bspline_order=3, random_displacement=None, rng=rng
        )
        reinserted_mods = insert_back_to_original_shape(deformed_mods, cropped_bbox, original_shape)
        reinserted_mask = insert_back_to_original_shape(deformed_mask, cropped_bbox, original_shape)
        tcs_segs_mods[int(label_)] = [reinserted_mask, reinserted_mods]
    relocated_mask, relocated_modalities = rearrange_subcomponents_into_core(tcs_segs_mods, original_tumor_core, rng=rng)
    filled_modalities, filled_mask = inpaint(
        relocated_modalities, relocated_mask, np.unique(relocated_mask)[1:], original_tumor_core, threshold=10, sigma=1.0, rng=rng
    )
    new_full_tumor_mods, new_full_tumor_mask = edema_management(
        filled_modalities, filled_mask, modalities, seg, original_tumor_core, labels, rng=rng
    )
    new_modalities = modalities.copy()
    tumor_mask_bool = new_full_tumor_mask > 0
    new_modalities[:, tumor_mask_bool] = new_full_tumor_mods[:, tumor_mask_bool]
    invert_mask = ~tumor_mask_bool
    for mod in range(new_modalities.shape[0]):
        smoothed = smooth_interface_with_kernel(
            new_modalities[mod], tumor_mask_bool, invert_mask,
            dilate_iter=3, alpha=0.7, kernel_size=(2, 2, 2)
        )
        if brain_mask is not None:
            bm = brain_mask
            new_modalities[mod] = smoothed * bm
            fill_val = float(smoothed.min())
            new_modalities[mod][~bm] = fill_val
        else:
            new_modalities[mod] = smoothed
    return new_modalities.astype(np.float32, copy=False), new_full_tumor_mask.astype(np.uint8, copy=False), affine, brain_mask

def augment(input_path: str, output_path: str, samples: int, normalize: bool = False,
            seed: int = None) -> None:
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed) 
    in_dir = Path(input_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dict: Dict[str, Dict[str, Optional[str]]] = {}
    for case_dir in sorted([p for p in in_dir.iterdir() if p.is_dir()]):
        case_name = case_dir.name
        case_paths = _discover_case(case_dir)
        cases_dict[case_name] = case_paths
    for case_name, paths in cases_dict.items():
        for i in range(int(samples)):
            print(f"[BT-CAP] {case_name}: generating augmented volume {i+1}/{samples}")
            new_modalities, new_segmentation, affine, brain_mask = process_case(paths, rng=rng)
            if normalize:
                new_modalities = np.stack([ZScoreNormalize(new_modalities[c]) for c in range(new_modalities.shape[0])], axis=0)
            t1n_img = np_to_sitk(new_modalities[0], affine)
            t1c_img = np_to_sitk(new_modalities[1], affine)
            t2w_img = np_to_sitk(new_modalities[2], affine)
            t2f_img = np_to_sitk(new_modalities[3], affine)
            seg_img = np_to_sitk(new_segmentation.astype(np.uint8) if brain_mask is None
                                 else (new_segmentation.astype(np.uint8) * brain_mask.astype(np.uint8)), affine)
            new_case_folder = out_dir / f"Aug{i}-{case_name}"
            new_case_folder.mkdir(parents=True, exist_ok=True)
            def _suffix(path_str: str) -> str:
                return Path(path_str).name
            sitk.WriteImage(t1n_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t1n_path'])}"))
            sitk.WriteImage(t1c_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t1c_path'])}"))
            sitk.WriteImage(t2w_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t2w_path'])}"))
            sitk.WriteImage(t2f_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t2f_path'])}"))
            sitk.WriteImage(seg_img,  str(new_case_folder / f"Aug{i}-{_suffix(paths['seg_path'])}"))

def main() -> None:
    parser = argparse.ArgumentParser(description="Run BT-CAP augmentation pipeline.")
    parser.add_argument('--input', type=str, required=True, help="Path to the directory containing case folders.")
    parser.add_argument('--output', type=str, required=True, help="Path to save augmented output data.")
    parser.add_argument('--samples', type=int, default=1, help="Number of augmented samples per case.")
    parser.add_argument('--normalize', action='store_true', help="Apply Z-score normalization to outputs.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()
    augment(args.input, args.output, args.samples, args.normalize, seed=args.seed)

if __name__ == "__main__":
    main()