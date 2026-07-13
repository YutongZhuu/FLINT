#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("image", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()

image = Image.open(args.image).convert("RGB")
w, h = image.size
scale = min(640 / w, 640 / h)
nw, nh = round(w * scale), round(h * scale)
try:
    bilinear = Image.Resampling.BILINEAR
except AttributeError:
    bilinear = Image.BILINEAR
resized = image.resize((nw, nh), bilinear)
canvas = Image.new("RGB", (640, 640), (114, 114, 114))
left, top = (640 - nw) // 2, (640 - nh) // 2
canvas.paste(resized, (left, top))
array = np.asarray(canvas, dtype=np.float32) / 255.0
array.transpose(2, 0, 1)[None].tofile(args.output)
print(f"original={w}x{h} resized={nw}x{nh} pad=({left},{top}) scale={scale}")
