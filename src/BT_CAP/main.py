import os
import sys
import time
import math
import psutil
import argparse
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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


def _prepare_threads_per_process():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        pass

def _estimate_job_memory(paths_dict: Dict[str, str]) -> int:
    modalities, seg, brain_mask, _ = load_modalities_and_mask(paths_dict)
    total_bytes = modalities.nbytes + seg.nbytes
    if brain_mask is not None:
        total_bytes += brain_mask.nbytes
    return int(total_bytes * 1.5)

def _estimate_safe_workers(input_cases: List[Path],
                           requested_workers: Optional[int]
                          ) -> int:
    cpu_count = multiprocessing.cpu_count()
    desired = requested_workers or cpu_count
    try:
        first_case_paths = _discover_case(input_cases[0])
        job_mem = _estimate_job_memory(first_case_paths)
    except Exception:
        job_mem = 500 * 1024**2
    total_ram = psutil.virtual_memory().total
    safe_jobs = max(1, math.floor((total_ram * 0.75) / job_mem))
    max_workers = min(desired, safe_jobs, cpu_count)
    return max_workers

def process_case(paths_dict: Dict[str, str],
                 queue,
                 case_name: str,
                 sample_idx: int,
                 rng: np.random.Generator,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
    relocated_mask, relocated_modalities = rearrange_subcomponents_into_core(
        tcs_segs_mods, original_tumor_core, case_name, sample_idx, queue, rng=rng
    )
    filled_modalities, filled_mask = inpaint(
        relocated_modalities, relocated_mask, np.unique(relocated_mask)[1:], original_tumor_core,
        threshold=10, sigma=1.0, rng=rng
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
    return (new_modalities.astype(np.float32, copy=False),
            new_full_tumor_mask.astype(np.uint8, copy=False),
            affine,
            brain_mask)

def _write_outputs(case_name: str,
                   i: int,
                   paths: Dict[str, str],
                   out_dir: Path,
                   new_modalities: np.ndarray,
                   new_segmentation: np.ndarray,
                   affine: np.ndarray,
                   brain_mask: Optional[np.ndarray]) -> None:
    new_case_folder = out_dir / f"Aug{i}-{case_name}"
    new_case_folder.mkdir(parents=True, exist_ok=True)
    def _suffix(path_str: str) -> str:
        return Path(path_str).name
    t1n_img = np_to_sitk(new_modalities[0], affine)
    t1c_img = np_to_sitk(new_modalities[1], affine)
    t2w_img = np_to_sitk(new_modalities[2], affine)
    t2f_img = np_to_sitk(new_modalities[3], affine)
    seg_img = np_to_sitk(
        new_segmentation.astype(np.uint8) if brain_mask is None
        else (new_segmentation.astype(np.uint8) * brain_mask.astype(np.uint8)),
        affine
    )
    sitk.WriteImage(t1n_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t1n_path'])}"))
    sitk.WriteImage(t1c_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t1c_path'])}"))
    sitk.WriteImage(t2w_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t2w_path'])}"))
    sitk.WriteImage(t2f_img, str(new_case_folder / f"Aug{i}-{_suffix(paths['t2f_path'])}"))
    sitk.WriteImage(seg_img,  str(new_case_folder / f"Aug{i}-{_suffix(paths['seg_path'])}"))

def _stable_int_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def _job_seed(base_seed: Optional[int], case_name: str, sample_idx: int) -> Optional[int]:
    if base_seed is None:
        return None
    case_tag = _stable_int_from_str(case_name)
    ss = np.random.SeedSequence([base_seed, sample_idx, case_tag])
    return int(ss.generate_state(1)[0])

def _augment_one_sample_with_queue(case_name, case_dir, out_dir, sample_idx,
                                   normalize, base_seed, queue):
    try:
        job_seed = _job_seed(base_seed, case_name, sample_idx)
        rng = np.random.default_rng(job_seed) if job_seed is not None else np.random.default_rng()

        tag = _augment_one_sample(
            case_name=case_name,
            case_dir=case_dir,
            out_dir=out_dir,
            sample_idx=sample_idx,
            normalize=normalize,
            queue=queue,
            rng=rng
        )
        queue.put(("done", case_name, sample_idx, "success"))
        return tag
    except Exception as e:
        queue.put(("done", case_name, sample_idx, f"fail: {e}"))
        raise

def _augment_one_sample(case_name: str,
                        case_dir: str,
                        out_dir: str,
                        sample_idx: int,
                        normalize: bool,
                        queue,
                        rng: np.random.Generator) -> str:
    _prepare_threads_per_process()
    paths = _discover_case(Path(case_dir))
    new_modalities, new_segmentation, affine, brain_mask = process_case(
        paths, queue, case_name, sample_idx, rng=rng
    )
    if normalize:
        new_modalities = np.stack(
            [ZScoreNormalize(new_modalities[c]) for c in range(new_modalities.shape[0])],
            axis=0
        )
    _write_outputs(case_name, sample_idx, paths, Path(out_dir),
                   new_modalities, new_segmentation, affine, brain_mask)
    return f"{case_name} (sample {sample_idx})"

def augment(input_path: str,
            output_path: str,
            samples: int,
            normalize: bool = False,
            seed: Optional[int] = None,
            workers: Optional[int] = None) -> None:
    in_dir = Path(input_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs: List[Path] = [p for p in sorted(in_dir.iterdir()) if p.is_dir()]
    if not case_dirs:
        raise RuntimeError(f"No case folders found in: {in_dir}")
    max_workers = _estimate_safe_workers(case_dirs, workers)
    print(f"[BT-CAP] Found {len(case_dirs)} case(s). "
          f"Generating {samples} sample(s) per case with up to {max_workers} worker(s).")
    total_jobs = len(case_dirs) * samples
    global_pbar = tqdm(total=total_jobs, desc="Total progress", position=0, colour='green')
    with multiprocessing.Manager() as manager:
        queue = manager.Queue(maxsize=10000)
        sample_pbars: Dict[tuple[str, int], tqdm] = {}
        next_position = 1
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for case_dir in case_dirs:
                for i in range(samples):
                    f = ex.submit(
                        _augment_one_sample_with_queue,
                        case_dir.name,
                        str(case_dir),
                        str(out_dir),
                        i,
                        normalize,
                        seed,
                        queue
                    )
                    futures.append(f)

            finished_jobs = 0
            while finished_jobs < total_jobs:
                try:
                    msg_type, case_name, sample_idx, *rest = queue.get(timeout=0.2)
                    sample_key = (case_name, sample_idx)

                    if msg_type == "island_placement":
                        done, total = rest
                        if sample_key not in sample_pbars:
                            sample_pbars[sample_key] = tqdm(
                                total=total,
                                desc=f"{case_name} sample {sample_idx}",
                                unit="step",
                                position=next_position,
                                leave=True,
                                ascii="░▒█"
                            )
                            next_position += 1
                        bar = sample_pbars[sample_key]
                        bar.n = done
                        bar.refresh()
                    elif msg_type == "done":
                        global_pbar.update(1) 
                        finished_jobs += 1
                except Exception:
                    time.sleep(0.1)
            for f in futures:
                f.result()
        manager.shutdown()
    global_pbar.close()
    for bar in list(sample_pbars.values()):
        bar.close()
    print("[BT-CAP] All augmentations completed.")
    sys.stdout.flush()

def main() -> None:
    parser = argparse.ArgumentParser(description="Run BT-CAP augmentation pipeline.")
    parser.add_argument('--input', type=str, required=True, help="Path to the directory containing case folders.")
    parser.add_argument('--output', type=str, required=True, help="Path to save augmented output data.")
    parser.add_argument('--samples', type=int, default=1, help="Number of augmented samples per case.")
    parser.add_argument('--normalize', action='store_true', help="Apply Z-score normalization to outputs.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--workers', type=int, default=None, help="Number of parallel workers (default: CPU count).")
    args = parser.parse_args()
    augment(args.input, args.output, args.samples, args.normalize, seed=args.seed, workers=args.workers)

if __name__ == "__main__":
    main()