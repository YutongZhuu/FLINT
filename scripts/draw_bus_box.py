#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LINE = re.compile(
    r"^(?P<label>\S+)\s+(?P<score>[0-9.]+)\s+"
    r"box=\[(?P<x1>[0-9.]+),\s*(?P<y1>[0-9.]+),\s*"
    r"(?P<x2>[0-9.]+),\s*(?P<y2>[0-9.]+)\]$"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the highest-confidence bus detection in red."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    buses = []
    for line in args.detections.read_text().splitlines():
        match = LINE.match(line.strip())
        if match and match["label"] == "bus":
            buses.append(
                (
                    float(match["score"]),
                    tuple(float(match[name]) for name in ("x1", "y1", "x2", "y2")),
                )
            )
    if not buses:
        raise SystemExit("no bus detection found")

    score, box = max(buses)
    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 160))
    draw.rectangle(box, outline="red", width=width)

    text = f"bus {score:.2f}"
    font = ImageFont.load_default(size=max(14, round(min(image.size) / 35)))
    text_box = draw.textbbox((box[0], box[1]), text, font=font, stroke_width=1)
    text_height = text_box[3] - text_box[1]
    label_y = max(0, box[1] - text_height - 6)
    label_box = draw.textbbox((box[0], label_y), text, font=font, stroke_width=1)
    draw.rectangle(
        (label_box[0] - 3, label_box[1] - 3, label_box[2] + 3, label_box[3] + 3),
        fill="red",
    )
    draw.text((box[0], label_y), text, fill="white", font=font, stroke_width=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"wrote {args.output}: bus score={score:.4f} box={box}")


if __name__ == "__main__":
    main()
