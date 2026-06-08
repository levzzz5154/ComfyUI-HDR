#!/usr/bin/env python3
"""Convert .npy latent file to 16-bit PNG or 10-bit HDR AVIF."""

import sys
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import safetensors.torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image


def load_npy(filepath):
    arr = np.load(filepath)
    print(f"Loaded: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def load_latent(filepath):
    latent = safetensors.torch.load_file(filepath, device="cpu")
    arr = latent["latent_tensor"].float().numpy()
    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215
    arr = arr * multiplier
    print(f"Loaded: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def tonemap(arr, mode="none", gamma=2.2):
    if mode == "none":
        return arr
    elif mode == "linear_to_srgb":
        return np.power(np.clip(arr, 0, None), 1.0 / gamma)
    elif mode == "srgb_to_linear":
        return np.power(np.clip(arr, 0, 1), gamma)
    elif mode == "reinhard":
        return arr / (arr + 1.0)
    elif mode == "aces":
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((arr * (a * arr + b)) / (arr * (c * arr + d) + e), 0, 1)
    else:
        return arr


def prepare_array(arr):
    if arr.ndim == 4:
        if arr.shape[3] <= 4:
            arr = arr[0]
        else:
            arr = arr[0].transpose(1, 2, 0)
    elif arr.ndim == 3:
        if arr.shape[0] <= 4:
            arr = arr.transpose(1, 2, 0)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr


def normalize_to_16bit(arr, tonemap_mode="none", gamma=2.2):
    arr = prepare_array(arr)

    if tonemap_mode != "none":
        print(f"Tonemap: {tonemap_mode}, gamma={gamma}")
        arr = tonemap(arr, tonemap_mode, gamma)

    min_val = arr.min()
    max_val = arr.max()
    print(f"Min: {min_val}, Max: {max_val}")
    arr = (arr - min_val) / (max_val - min_val) * 65535
    arr = arr.astype(np.uint16)

    return arr


def save_16bit_png(arr, output_path):
    if HAS_CV2:
        cv2.imwrite(str(output_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(arr, mode='I;16').save(output_path)
    print(f"Saved: {output_path}")


def save_hdr_avif(arr, output_path, peak_luminance=1000, quality=90):
    arr = prepare_array(arr)

    min_val = arr.min()
    max_val = arr.max()
    print(f"Min: {min_val}, Max: {max_val}")

    arr = (arr - min_val) / (max_val - min_val) * 65535
    arr = arr.astype(np.uint16)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_png = tmp.name

    try:
        if HAS_CV2:
            cv2.imwrite(tmp_png, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        else:
            Image.fromarray(arr, mode='I;16').save(tmp_png)

        max_cll = int(peak_luminance)
        max_fall = int(peak_luminance * 0.5)

        cmd = [
            'avifenc',
            '-d', '10',
            '--cicp', '9/16/9',
            '--clli', f'{max_cll},{max_fall}',
            '-q', str(quality),
            '-s', '6',
            tmp_png,
            str(output_path)
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False

        print(f"Saved HDR AVIF: {output_path} (peak: {peak_luminance} nits)")
        return True
    finally:
        if os.path.exists(tmp_png):
            os.remove(tmp_png)


def main():
    parser = argparse.ArgumentParser(description="Convert .npy or .latent file to 16-bit PNG or 10-bit HDR AVIF")
    parser.add_argument("input", help="Input .npy or .latent file")
    parser.add_argument("output", nargs="?", help="Output file (default: input name with extension)")
    parser.add_argument("-f", "--format", choices=["png", "avif"], default="png",
                        help="Output format (default: png)")
    parser.add_argument("-t", "--tonemap", choices=["none", "linear_to_srgb", "srgb_to_linear", "reinhard", "aces"],
                        default="none", help="Tonemap mode for PNG (default: none)")
    parser.add_argument("-g", "--gamma", type=float, default=2.2, help="Gamma value (default: 2.2)")
    parser.add_argument("-p", "--peak", type=float, default=1000,
                        help="Peak luminance for HDR AVIF in nits (default: 1000)")
    parser.add_argument("-q", "--quality", type=int, default=90,
                        help="AVIF quality 0-100 (default: 90)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(f'.{args.format}')

    if input_path.suffix == ".latent":
        arr = load_latent(input_path)
    else:
        arr = load_npy(input_path)

    if args.format == "avif":
        save_hdr_avif(arr, output_path, args.peak, args.quality)
    else:
        arr_16bit = normalize_to_16bit(arr, args.tonemap, args.gamma)
        save_16bit_png(arr_16bit, output_path)


if __name__ == '__main__':
    main()
