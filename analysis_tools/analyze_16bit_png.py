#!/usr/bin/env python3
"""Analyze 16-bit PNG color channels for quantization and bit depth."""

import sys
from pathlib import Path
from math import log2, gcd
from functools import reduce

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_quantization_step(arr):
    sorted_vals = np.sort(np.unique(arr))
    if len(sorted_vals) < 2:
        return 0
    diffs = np.diff(sorted_vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0
    return reduce(gcd, diffs.astype(int))


def analyze_channel(data, name, color):
    unique_vals = np.unique(data)
    unique_count = len(unique_vals)
    theoretical_max = 65536

    min_val = int(np.min(data))
    max_val = int(np.max(data))
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))

    effective_bits = log2(unique_count) if unique_count > 0 else 0

    quant_step = find_quantization_step(data)

    if quant_step > 1:
        if quant_step == 256:
            quant_type = "8-bit quantized (step=256)"
        elif quant_step == 64:
            quant_type = "10-bit quantized (step=64)"
        elif quant_step == 16:
            quant_type = "12-bit quantized (step=16)"
        elif quant_step == 4:
            quant_type = "14-bit quantized (step=4)"
        else:
            quant_type = f"Quantized (step={quant_step})"
    else:
        quant_type = "Full 16-bit (no obvious quantization)"

    stats = {
        'name': name,
        'unique_count': unique_count,
        'unique_ratio': unique_count / theoretical_max,
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'effective_bits': effective_bits,
        'quant_step': quant_step,
        'quant_type': quant_type,
        'data': data,
        'unique_vals': unique_vals,
        'color': color
    }

    return stats


def print_stats(stats):
    print(f"\n{stats['name']} Channel:")
    print(f"  Range: {stats['min']} - {stats['max']}")
    print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f"  Unique values: {stats['unique_count']} / 65536 ({stats['unique_ratio']*100:.2f}%)")
    print(f"  Effective bit depth: {stats['effective_bits']:.2f} bits")
    print(f"  Quantization: {stats['quant_type']}")


def load_16bit_image(filepath):
    if HAS_CV2:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {filepath}")

        print(f"Loaded with OpenCV: shape={img.shape}, dtype={img.dtype}")

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        img = Image.open(filepath)
        print(f"Loaded with PIL: mode={img.mode}, size={img.size}")

        if img.mode in ['I;16', 'I;16B', 'I']:
            arr = np.array(img, dtype=np.uint16)
            return arr, arr, arr
        elif img.mode in ['RGB;16', 'RGBA;16']:
            arr = np.array(img, dtype=np.uint16)
            if len(arr.shape) == 3:
                return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            return arr, arr, arr
        else:
            arr = np.array(img)
            if arr.dtype == np.uint8:
                print("Warning: Image is 8-bit")
                arr = arr.astype(np.uint16) * 256
            if len(arr.shape) == 3:
                if arr.shape[2] >= 3:
                    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            return arr, arr, arr


def analyze_png(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return

    print(f"Analyzing: {filepath}")

    r, g, b = load_16bit_image(filepath)

    print(f"\nData type: {r.dtype}")
    print(f"Channel data range: {r.min()} - {r.max()}")

    r_stats = analyze_channel(r, 'Red', 'red')
    g_stats = analyze_channel(g, 'Green', 'green')
    b_stats = analyze_channel(b, 'Blue', 'blue')

    for stats in [r_stats, g_stats, b_stats]:
        print_stats(stats)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for ax, stats in zip(axes, [r_stats, g_stats, b_stats]):
        data_flat = stats['data'].flatten()

        if stats['unique_count'] > 100:
            bins = min(256, stats['unique_count'] // 10)
        else:
            bins = stats['unique_count']

        ax.hist(data_flat, bins=bins, color=stats['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{stats['name']} Channel - {stats['unique_count']} unique values, {stats['effective_bits']:.2f} effective bits\n{stats['quant_type']}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath.parent / f"{filepath.stem}_analysis.png", dpi=150)
    plt.close()

    print(f"\nHistogram saved to: {filepath.parent / f'{filepath.stem}_analysis.png'}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = Path(__file__).parent.parent / "ComfyUI_00008_.png"

    analyze_png(filepath)
