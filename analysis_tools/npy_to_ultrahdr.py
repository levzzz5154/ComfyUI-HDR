#!/usr/bin/env python3
"""Convert .npy latent file to Ultra HDR JPEG with gain map."""

import sys
import argparse
import struct
import io
from pathlib import Path
import numpy as np

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


def create_sdr_image(arr, gamma=2.2):
    min_val = arr.min()
    max_val = arr.max()
    
    normalized = (arr - min_val) / (max_val - min_val)
    
    sdr = np.power(normalized, 1.0 / gamma)
    sdr = np.clip(sdr, 0, 1)
    
    sdr_8bit = (sdr * 255).astype(np.uint8)
    return sdr_8bit, normalized, min_val, max_val


def create_gain_map(hdr_normalized, sdr_8bit, gamma=2.2, max_boost=3.0):
    sdr_linear = sdr_8bit.astype(np.float32) / 255.0
    
    sdr_lum = np.mean(sdr_linear, axis=2)
    
    bright_mask = sdr_lum > 0.75
    
    ratio = np.ones_like(sdr_lum) * 0.4
    
    ratio[bright_mask] = 1.0 + (sdr_lum[bright_mask] - 0.75) * 8.0
    ratio = np.clip(ratio, 0.33, max_boost)
    
    log_gain = np.log2(ratio)
    
    log_max = np.log2(max_boost)
    normalized_gain = (log_gain) / (2 * log_max) + 0.5
    normalized_gain = np.clip(normalized_gain, 0, 1)
    
    gain_map_8bit = (normalized_gain * 255).astype(np.uint8)
    
    return gain_map_8bit


def create_ultrahdr_xmp(gain_map_length, max_boost=3.0):
    gain_map_max = np.log2(max_boost)
    xmp = f'''<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.5.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
    xmlns:Container="http://ns.google.com/photos/1.0/container/"
    xmlns:Item="http://ns.google.com/photos/1.0/container/item/"
    hdrgm:Version="1.0"
    hdrgm:GainMapMin="-{gain_map_max:.6f}"
    hdrgm:GainMapMax="{gain_map_max:.6f}"
    hdrgm:Gamma="1.0"
    hdrgm:OffsetSDR="0.0625"
    hdrgm:OffsetHDR="0.0625"
    hdrgm:HDRCapacityMin="0"
    hdrgm:HDRCapacityMax="{gain_map_max:.6f}">
   <Container:Directory>
    <rdf:Seq>
     <rdf:li rdf:parseType="Resource">
      <Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/>
     </rdf:li>
     <rdf:li rdf:parseType="Resource">
      <Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{gain_map_length}"/>
     </rdf:li>
    </rdf:Seq>
   </Container:Directory>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>'''
    return xmp


def create_mpf_marker(primary_length, gain_map_length):
    mpf_data = struct.pack('>IIIIII',
        0x4D504600,
        0x4D504600,
        0x00000030,
        0x00010000,
        0x00000001,
        0x00000000
    )
    mpf_data += struct.pack('>IIII',
        primary_length,
        0x00020000,
        len(mpf_data) + 24 + gain_map_length,
        0
    )
    return mpf_data


def save_ultrahdr_jpeg(sdr_8bit, gain_map_8bit, output_path, quality=95, max_boost=8.0):
    sdr_rgb = cv2.cvtColor(sdr_8bit, cv2.COLOR_RGB2BGR) if HAS_CV2 else sdr_8bit
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, sdr_encoded = cv2.imencode('.jpg', sdr_rgb, encode_param)
    sdr_bytes = sdr_encoded.tobytes()
    
    if HAS_CV2:
        if len(gain_map_8bit.shape) == 2:
            gain_map_bgr = cv2.cvtColor(gain_map_8bit, cv2.COLOR_GRAY2BGR)
        else:
            gain_map_bgr = gain_map_8bit
        _, gain_encoded = cv2.imencode('.jpg', gain_map_bgr, encode_param)
        gain_bytes = gain_encoded.tobytes()
    else:
        gain_img = Image.fromarray(gain_map_8bit, mode='L')
        gain_buffer = io.BytesIO()
        gain_img.save(gain_buffer, format='JPEG', quality=quality)
        gain_bytes = gain_buffer.getvalue()
    
    xmp = create_ultrahdr_xmp(len(gain_bytes), max_boost)
    xmp_bytes = xmp.encode('utf-8')
    
    output = bytearray()
    
    soi_found = False
    app0_found = False
    pos = 0
    
    while pos < len(sdr_bytes):
        if sdr_bytes[pos] == 0xFF:
            marker = sdr_bytes[pos + 1] if pos + 1 < len(sdr_bytes) else 0
            
            if marker == 0xD8:
                output.extend(sdr_bytes[pos:pos+2])
                pos += 2
                soi_found = True
                continue
            elif marker == 0xE0:
                app0_end = pos + 2
                if pos + 4 <= len(sdr_bytes):
                    app0_len = struct.unpack('>H', sdr_bytes[pos+2:pos+4])[0]
                    app0_end = pos + 2 + app0_len
                output.extend(sdr_bytes[pos:app0_end])
                pos = app0_end
                app0_found = True
                
                output.extend(b'\xFF\xE1')
                xmp_header = b'http://ns.adobe.com/xap/1.0/\x00'
                xmp_segment = xmp_header + xmp_bytes
                output.extend(struct.pack('>H', len(xmp_segment) + 2))
                output.extend(xmp_segment)
                continue
            elif marker == 0xDA:
                output.extend(sdr_bytes[pos:])
                break
            else:
                seg_len = 0
                if pos + 4 <= len(sdr_bytes) and marker not in [0xD9]:
                    seg_len = struct.unpack('>H', sdr_bytes[pos+2:pos+4])[0]
                output.extend(sdr_bytes[pos:pos+2+seg_len])
                pos += 2 + seg_len
                continue
        pos += 1
    
    output.extend(gain_bytes)
    
    with open(output_path, 'wb') as f:
        f.write(output)
    
    print(f"Saved Ultra HDR JPEG: {output_path}")
    print(f"  SDR size: {len(sdr_bytes)} bytes")
    print(f"  Gain map size: {len(gain_bytes)} bytes")
    print(f"  Total size: {len(output)} bytes")


def main():
    parser = argparse.ArgumentParser(description="Convert .npy latent to Ultra HDR JPEG")
    parser.add_argument("input", help="Input .npy file")
    parser.add_argument("output", nargs="?", help="Output .jpg file (default: input name with .jpg)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("-g", "--gamma", type=float, default=2.2,
                        help="Gamma for SDR tonemap (default: 2.2)")
    parser.add_argument("-b", "--max-boost", type=float, default=3.0,
                        help="Maximum HDR boost factor (default: 3.0)")
    args = parser.parse_args()
    
    if not HAS_CV2:
        print("Error: OpenCV (cv2) is required for Ultra HDR encoding")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.jpg')
    
    arr = load_npy(input_path)
    arr = prepare_array(arr)
    
    print(f"Creating SDR image (gamma={args.gamma})...")
    sdr_8bit, hdr_normalized, min_val, max_val = create_sdr_image(arr, args.gamma)
    print(f"Data range: {min_val:.4f} to {max_val:.4f}")
    
    print(f"Creating gain map (max_boost={args.max_boost})...")
    gain_map_8bit = create_gain_map(hdr_normalized, sdr_8bit, args.gamma, args.max_boost)
    
    save_ultrahdr_jpeg(sdr_8bit, gain_map_8bit, output_path, args.quality, args.max_boost)


if __name__ == '__main__':
    main()
