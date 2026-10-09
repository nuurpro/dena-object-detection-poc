"""Reproduce the visually reviewed PoC selection from the saved contact index.

Indices were selected after viewing all 17 contact sheets, then revised with user
authorization when the archive changed from 476 to 175 images. They are not image
labels: posters may include real packages and distribution filenames are noisy.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED = [
    4, 5, 7, 8, 9, 10, 11, 13,
    14, 16, 18, 19, 20, 21, 22, 23, 25, 30, 31, 32, 34, 35,
    38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 53, 54,
    55, 58, 60, 61,
    73, 74, 75, 76, 77, 80, 86, 87, 88, 90,
    110, 111, 114, 115, 117, 119, 120, 122, 123, 125, 126,
    135, 136, 139, 141, 142,
    151, 153, 155, 156, 157, 158, 160,
    163, 166, 169, 170, 171, 172, 176,
    180, 184, 187, 194, 197,
    200, 201, 202, 213, 214,
    218, 219, 223,
    240, 246, 248,
    259, 261, 266,
    271,
]
BENCHMARK_TIMESTAMPS = {
    "1790078345789": "5303", "1790068813143": "4647",
    "1790064417011": "5140", "1790074129177": "9615",
    "1790066200804": "7663",
}

def main():
    with (ROOT / "outputs/selection_review/contact_index.csv").open(encoding="utf-8", newline="") as stream:
        indexed = {int(row["index"]): row for row in csv.DictReader(stream)}
    assert len(SELECTED) == len(set(SELECTED)) == 100
    source = ROOT.parent / "Archive_1790087449"
    for index in SELECTED:
        if not (source / indexed[index]["source_path"]).is_file():
            raise ValueError(f"Reviewed source no longer exists: {indexed[index]['source_path']}")
    output = ROOT / "configs/selection.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_path", "outlet_id", "benchmark_id", "review_status", "contact_index", "selection_reason"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in SELECTED:
            row = indexed[index]
            path = row["source_path"]
            outlet = re.search(r"img-[a-zA-Z0-9]+_(\d+)-\d+", path)
            if not outlet:
                raise ValueError(f"Unresolved outlet: {path}")
            benchmark = next((value for timestamp, value in BENCHMARK_TIMESTAMPS.items() if path.endswith(timestamp + ".jpg")), "")
            reason = {
                "shelf": "Shelf/refrigerator scene selected for scale, angle, glare or clutter diversity",
                "no_distribution": "Visually reviewed competitor/mixed scene; filename is not ground truth",
                "poster": "Printed DENA artwork/context to test package-versus-artwork confusion",
            }[row["category"]]
            writer.writerow(dict(source_path=path, outlet_id=outlet.group(1), benchmark_id=benchmark,
                                 review_status="reviewed", contact_index=index, selection_reason=reason))
    print(f"Saved 100 visually reviewed selection rows to {output}")

if __name__ == "__main__":
    main()
