#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSAS-ViT test script for RMSE evaluation.

This script evaluates the FSAS + ViT model on test data and computes:
- Wavefront RMSE (if Zseg is provided)
- Coefficient RMSE (full 210 elements)
- Coefficient RMSE (74 valid coefficients for 77-coefficient setup)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed


CURRENT_DIR = Path(__file__).resolve().parent
CURRENT_MODEL_PATH = CURRENT_DIR / "psf_zernike_model_fcas_vit.py"


def load_input_npy(path: str, expected_num_images: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single `.npy` sample."""
    data = np.load(path, allow_pickle=True)
    inputs = data[0][:expected_num_images]
    coeffs_gt = data[1]["gt_a"] if isinstance(data[1], dict) else data[1]
    return inputs, coeffs_gt


def load_python_module(module_path: Path, module_name: str):
    """Import a Python module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    """Handle common checkpoint wrappers."""
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)!r}")
    return checkpoint_obj


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize DataParallel checkpoints to bare keys."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def infer_num_images(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer input channel count from patch embedding weights."""
    candidate_keys = (
        "backbone.patch_embed.proj.weight",
        "patch_embed.proj.weight",
    )
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 4:
            return int(tensor.shape[1])
    raise KeyError("Unable to infer num_images from checkpoint.")


def infer_num_coefficients(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer regression output dimension from the final head layer."""
    preferred_keys = (
        "head.7.weight",
        "head.3.weight",
        "head.weight",
    )
    for key in preferred_keys:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])

    head_candidates = []
    for key, tensor in state_dict.items():
        if not key.startswith("head.") or not key.endswith(".weight") or tensor.ndim != 2:
            continue
        parts = key.split(".")
        try:
            layer_index = int(parts[1])
        except (IndexError, ValueError):
            continue
        head_candidates.append((layer_index, int(tensor.shape[0])))

    if head_candidates:
        head_candidates.sort()
        return head_candidates[-1][1]

    raise KeyError("Unable to infer num_coefficients from checkpoint.")


def infer_variant(state_dict: Dict[str, torch.Tensor]) -> str:
    """Infer ViT variant from embedding dimension."""
    tensor = state_dict.get("backbone.patch_embed.proj.weight")

    if tensor is None or tensor.ndim != 4:
        raise KeyError("Unable to infer variant from checkpoint.")

    embed_dim = int(tensor.shape[0])
    variant_map = {
        192: "tiny",
        384: "small",
        768: "base",
        1024: "large",
    }
    if embed_dim not in variant_map:
        raise ValueError(f"Unsupported embedding dim in checkpoint: {embed_dim}")
    return variant_map[embed_dim]


def load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load and normalize state dict keys."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    return strip_module_prefix(extract_state_dict(checkpoint_obj))


def build_model_from_checkpoint(state_dict: Dict[str, torch.Tensor], args) -> Tuple[nn.Module, Dict[str, object]]:
    """Build the correct model implementation for the given checkpoint."""
    variant = args.variant or infer_variant(state_dict)
    num_images = args.num_images or infer_num_images(state_dict)
    num_coefficients = args.num_coefficients or infer_num_coefficients(state_dict)

    module = load_python_module(CURRENT_MODEL_PATH, "fcas_vit_model")
    model = module.create_model(
        input_size=args.img_size,
        patch_size=args.patch_size,
        num_images=num_images,
        channels_per_image=1,
        num_coefficients=num_coefficients,
        variant=variant,
        pretrained=False,
        dropout=0.0,
        use_vit_module=args.use_vit_module,
        fsas_patch_size=args.fsas_patch_size,
        cbam_reduction=args.cbam_reduction,
        cbam_kernel_size=args.cbam_kernel_size,
        ca_reduction=args.ca_reduction,
    )
    metadata = {
        "variant": variant,
        "num_images": num_images,
        "num_coefficients": num_coefficients,
    }
    return model, metadata


def load_model_weights(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Load weights, adapting to DataParallel wrappers if needed."""
    model_is_parallel = isinstance(model, nn.DataParallel)

    if model_is_parallel:
        adapted_state = {f"module.{key}": value for key, value in state_dict.items()}
    else:
        adapted_state = state_dict

    model.load_state_dict(adapted_state, strict=True)
    return model


def process_1th_zernike_coef(coef: np.ndarray) -> np.ndarray:
    """Normalize the piston term."""
    coef = coef.copy()
    c0 = coef[..., 0]
    n = np.round(c0)
    coef[..., 0] = c0 - n
    return coef


def wavefront_rmse(coef_pre: np.ndarray, coef_gt: np.ndarray, zseg: np.ndarray) -> float:
    """Compute wavefront RMSE."""
    coef_pre = process_1th_zernike_coef(coef_pre)
    coef_gt = process_1th_zernike_coef(coef_gt)
    coef_pre_reshaped = coef_pre[:, :, None, None]
    coef_gt_reshaped = coef_gt[:, :, None, None]
    result_pre = (zseg * coef_pre_reshaped).sum(axis=1).sum(axis=0)
    result_gt = (zseg * coef_gt_reshaped).sum(axis=1).sum(axis=0)
    return float(np.sqrt(np.mean((result_pre - result_gt) ** 2)))


def coef_rmse_full(coef_pre: np.ndarray, coef_gt: np.ndarray) -> float:
    """Compute full-coefficient RMSE on the reconstructed [7, 30] grid."""
    coef_pre = process_1th_zernike_coef(coef_pre)
    coef_gt = process_1th_zernike_coef(coef_gt)
    return float(np.sqrt(np.mean((coef_pre - coef_gt) ** 2)))


def coef74_rmse(coef_pre: np.ndarray, coef_gt: np.ndarray) -> float:
    """Compute the 74-valid-coefficients RMSE for the 77-coefficient setup."""
    coef_pre = process_1th_zernike_coef(coef_pre)
    coef_gt = process_1th_zernike_coef(coef_gt)
    pre_77 = coef_pre[:, :11].reshape(-1)
    gt_77 = coef_gt[:, :11].reshape(-1)
    mask = np.ones(77, dtype=bool)
    mask[66:69] = False
    return float(np.sqrt(np.mean((pre_77[mask] - gt_77[mask]) ** 2)))


def coeffs77_to_full(coeffs_flat: np.ndarray) -> np.ndarray:
    """Convert 77 coefficients to the full [7, 30] layout."""
    full = np.zeros((7, 30), dtype=np.float32)
    full[:, :11] = coeffs_flat.reshape(7, 11)
    return full


def coeffs25_to_full(coeffs_flat: np.ndarray) -> np.ndarray:
    """Convert 25 coefficients to the legacy sparse [7, 30] layout."""
    full = np.zeros((7, 30), dtype=np.float32)
    full[0, 3] = coeffs_flat[0]
    full[1:7, :4] = coeffs_flat[1:].reshape(6, 4)
    return full


def coeffs_to_full(coeffs_flat: np.ndarray, num_coefficients: int) -> np.ndarray:
    """Map model outputs back to the [7, 30] coefficient grid."""
    if num_coefficients == 77:
        return coeffs77_to_full(coeffs_flat)
    if num_coefficients == 25:
        return coeffs25_to_full(coeffs_flat)
    raise ValueError(f"Unsupported num_coefficients: {num_coefficients}")


def process_single_file(args_tuple):
    """Run inference and metric calculation for one file."""
    (
        input_file,
        filename,
        model,
        device,
        zseg,
        model_lock,
        expected_num_images,
        num_coefficients,
    ) = args_tuple
    try:
        inputs, coeffs_gt = load_input_npy(input_file, expected_num_images)
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        if inputs.ndim != 3 or inputs.shape[0] != expected_num_images:
            print(f"Skipping invalid input {input_file}: shape {inputs.shape}")
            return None

        input_tensor = torch.from_numpy(inputs).unsqueeze(0).to(device)

        with model_lock:
            with torch.no_grad():
                outputs = model(input_tensor)

        coeffs_flat = outputs.detach().cpu().numpy().reshape(-1)
        coeffs_pre = coeffs_to_full(coeffs_flat, num_coefficients)

        wave_rmse = wavefront_rmse(coeffs_pre, coeffs_gt, zseg) if zseg is not None else 0.0
        coef_full = coef_rmse_full(coeffs_pre, coeffs_gt)

        if num_coefficients == 77:
            coef74 = coef74_rmse(coeffs_pre, coeffs_gt)
            print(
                f"Processed: {filename} - Wavefront: {wave_rmse:.6f}, "
                f"Coef(full): {coef_full:.6f}, Coef74: {coef74:.6f}"
            )
            return (filename, wave_rmse, coef_full, coef74)

        print(
            f"Processed: {filename} - Wavefront: {wave_rmse:.6f}, "
            f"Coef(full): {coef_full:.6f}"
        )
        return (filename, wave_rmse, coef_full, float("nan"))
    except Exception as exc:
        print(f"Error processing {filename}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="FSAS-ViT RMSE evaluation")
    parser.add_argument("--input_path", type=str, required=True, help="Test file or directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--Zseg_path", type=str, default=None, help="Path to Zseg.npy")
    parser.add_argument("--output_dir", type=str, default="./test_results_fcas_vit", help="Output directory")
    parser.add_argument("--num_threads", type=int, default=32, help="Worker threads")
    parser.add_argument(
        "--variant",
        "--vit_variant",
        dest="variant",
        type=str,
        default=None,
        choices=["tiny", "small", "base", "large"],
        help="ViT variant, default: infer from checkpoint",
    )
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--fsas_patch_size", type=int, default=8, help="FSAS patch size")
    parser.add_argument("--cbam_reduction", type=int, default=16, help="CBAM channel reduction ratio")
    parser.add_argument("--cbam_kernel_size", type=int, default=7, help="CBAM spatial attention kernel size")
    parser.add_argument("--ca_reduction", type=int, default=32, help="CA (Coordinate Attention) channel reduction ratio")
    parser.add_argument("--img_size", type=int, default=112, help="Input image size")
    parser.add_argument("--num_images", type=int, default=None, help="Override input image count")
    parser.add_argument("--num_coefficients", type=int, default=None, help="Override output dimension")
    parser.add_argument("--use_vit_module", action="store_true", default=True, help="Use ViT backbone")
    parser.add_argument("--use_dataparallel", action="store_true", help="Wrap model with DataParallel")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    state_dict = load_checkpoint_state(args.checkpoint_path)
    model, metadata = build_model_from_checkpoint(state_dict, args)

    print(
        "Detected checkpoint: "
        f"variant={metadata['variant']}, "
        f"num_images={metadata['num_images']}, num_coefficients={metadata['num_coefficients']}"
    )

    if args.use_dataparallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    model = load_model_weights(model, state_dict)
    model.eval()
    print(f"Model loaded from: {args.checkpoint_path}")

    zseg = None
    if args.Zseg_path and os.path.exists(args.Zseg_path):
        zseg = np.load(args.Zseg_path, allow_pickle=True)
        print(f"Zseg loaded, shape: {zseg.shape}")
    else:
        print("Zseg not provided, Wavefront RMSE will be reported as 0")

    expected_num_images = int(metadata["num_images"])
    num_coefficients = int(metadata["num_coefficients"])

    if os.path.isfile(args.input_path):
        print(f"\n=== Testing single file: {args.input_path} ===")
        inputs, coeffs_gt = load_input_npy(args.input_path, expected_num_images)
        print(f"inputs shape: {inputs.shape}, coeffs_gt shape: {coeffs_gt.shape}")

        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)

        input_tensor = torch.from_numpy(inputs).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)

        coeffs_flat = outputs.detach().cpu().numpy().reshape(-1)
        coeffs_pre = coeffs_to_full(coeffs_flat, num_coefficients)

        wave_rmse = wavefront_rmse(coeffs_pre, coeffs_gt, zseg) if zseg is not None else 0.0
        coef_full = coef_rmse_full(coeffs_pre, coeffs_gt)

        print("\n==== Results ====")
        print(f"Wavefront RMSE:    {wave_rmse:.6f}")
        print(f"Coef RMSE (full):  {coef_full:.6f}  (210 elements)")
        if num_coefficients == 77:
            coef74 = coef74_rmse(coeffs_pre, coeffs_gt)
            print(f"Coef74 RMSE:       {coef74:.6f}  (74 valid coefficients)")
        return

    if not os.path.isdir(args.input_path):
        raise ValueError(f"input_path is neither file nor directory: {args.input_path}")

    print(f"\n=== Testing directory: {args.input_path} ===")
    os.makedirs(args.output_dir, exist_ok=True)

    file_list = [
        (os.path.join(args.input_path, filename), filename)
        for filename in os.listdir(args.input_path)
        if filename.endswith(".npy") and not filename.startswith("Zseg")
    ]
    print(f"Found {len(file_list)} files")

    if not file_list:
        print("No files to process")
        return

    model_lock = Lock()
    process_args = [
        (
            file_path,
            filename,
            model,
            device,
            zseg,
            model_lock,
            expected_num_images,
            num_coefficients,
        )
        for file_path, filename in file_list
    ]

    rmse_list = []
    start = time.time()
    max_workers = min(args.num_threads, len(process_args))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, item) for item in process_args]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rmse_list.append(result)

    elapsed = time.time() - start
    print(f"\nProcessed {len(rmse_list)} files in {elapsed:.2f}s")

    if rmse_list:
        avg_wave = float(np.nanmean([item[1] for item in rmse_list]))
        avg_full = float(np.nanmean([item[2] for item in rmse_list]))
        print("\n==== Average Results ====")
        print(f"Average Wavefront RMSE:   {avg_wave:.6f}")
        print(f"Average Coef RMSE (full): {avg_full:.6f}  (210 elements)")
        if num_coefficients == 77:
            avg_74 = float(np.nanmean([item[3] for item in rmse_list]))
            print(f"Average Coef74 RMSE:      {avg_74:.6f}  (74 valid coefficients)")

    out_csv = os.path.join(args.output_dir, "rmse_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Filename", "Wavefront_RMSE", "Coef_RMSE_full", "Coef74_RMSE"])
        writer.writerows(sorted(rmse_list, key=lambda item: item[0]))
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
