"""Produce numbered contact sheets of pending images; never alter the dataset."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ambiguity_review"


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["images"] if r["annotation_status"] == "pending"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for start in range(0, len(records), 12):
        sheet = Image.new("RGB", (1600, 1710), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(records[start:start + 12]):
            x, y = (offset % 4) * 400, (offset // 4) * 570
            with Image.open(ROOT / record["image_path"]) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (392, 524))
                sheet.paste(thumb, (x + (400 - thumb.width) // 2, y + 40))
            benchmark = record["benchmark_id"] or ""
            draw.text((x + 5, y + 4), f"#{start + offset + 1} {record['split']} benchmark={benchmark}", fill="black")
            draw.text((x + 5, y + 20), f"{record['width']}x{record['height']}", fill="black")
        destination = OUT / f"pending_{start // 12 + 1:02}.jpg"
        sheet.save(destination, quality=95)
        print(destination)
    print(f"Pending images: {len(records)}")


if __name__ == "__main__":
    main()
