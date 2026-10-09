"""Generate numbered contact sheets for human/agent dataset selection review."""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Archive_1790087449"
OUT = ROOT / "dena_poc" / "outputs" / "selection_review"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in SOURCE.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda p: p.relative_to(SOURCE).as_posix())
    candidates = [p for p in files if any(s in p.name for s in ("полкаси", "Холодильник", "LIDER", "Серия ХО", "Нет дистрибьюции", "Плакат"))]
    with (OUT / "contact_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "source_path", "category"])
        writer.writeheader()
        for i, p in enumerate(candidates, 1):
            category = "poster" if "Плакат" in p.name else "no_distribution" if "Нет дистрибьюции" in p.name else "shelf"
            writer.writerow(dict(index=i, source_path=p.relative_to(SOURCE).as_posix(), category=category))
    cols, rows, width, height = 6, 3, 320, 426
    for start in range(0, len(candidates), cols * rows):
        sheet = Image.new("RGB", (cols * width, rows * height), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(candidates[start:start + cols * rows]):
            x, y = offset % cols * width, offset // cols * height
            with Image.open(path) as source_image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(source_image).convert("RGB"), (width - 8, height - 45))
                sheet.paste(thumb, (x + (width - thumb.width) // 2, y + 35))
            number = start + offset + 1
            draw.text((x + 8, y + 3), f"#{number} {path.parent.name}", fill="black")
            category = "POSTER" if "Плакат" in path.name else "NO DISTRIBUTION" if "Нет дистрибьюции" in path.name else "SHELF"
            draw.text((x + 8, y + 17), category, fill="black")
        sheet.save(OUT / f"sheet_{start // (cols * rows) + 1:02}.jpg", quality=92)
    print(f"Reviewed candidate pool: {len(candidates)} images; {OUT}")

if __name__ == "__main__":
    main()
