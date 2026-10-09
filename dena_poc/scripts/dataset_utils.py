"""Shared dataset contract, image inspection, and outlet-level allocation."""

import hashlib
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re
import stat

from PIL import Image


BENCHMARKS = {
    'MR1 TOHIR/1л. DENA полкаси - тугирлаганимдан кейинги холат-5303-img-d6_1236-1790078345789.jpg': '5303',
    'SR6 Ibrohim/DENA Холодильник -4647-img-c7_5013-1790068813143.jpg': '4647',
    'SR6 Ibrohim/DENA LIDER точкада-XML_IDimg-e3_5140-1790064417011.jpg': '5140',
    'CH-5 Зилола/1л. DENA полкаси - тугирлаганимдан кейинги холат-XML_IDimg-e8_9615-1790074129177.jpg': '9615',
    'CH-3 Muhayyo/1л. DENA полкаси - тугирлаганимдан кейинги холат-XML_IDimg-f1_7663-1790066200804.jpg': '7663',
}
RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SPLITS = tuple(RATIOS)
EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ANNOTATION_STATUSES = {'pending', 'reviewed_positive', 'reviewed_negative'}
OUTLET_RE = re.compile(r'img-[a-z0-9]+_(\d+)-', re.IGNORECASE)


class DatasetError(ValueError):
    """Invalid inputs or a dataset that cannot safely be published."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inferred_outlet(path):
    match = OUTLET_RE.search(PurePosixPath(path).name)
    return match.group(1) if match else ''


def is_link(path):
    """Include Windows junctions/reparse points, not just symbolic links."""
    if path.is_symlink():
        return True
    try:
        attrs = getattr(path.lstat(), 'st_file_attributes', 0)
        return bool(attrs & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))
    except FileNotFoundError:
        return False


def reject_link_ancestors(path):
    for candidate in (path, *path.parents):
        if is_link(candidate):
            raise DatasetError(f'Unsafe symlink or junction: {candidate}')


def contained_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise DatasetError('Relative path must be nonempty text')
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if ('\\' in relative or posix.is_absolute() or windows.is_absolute()
            or windows.drive or any(p in {'', '.', '..'} for p in relative.split('/'))):
        raise DatasetError(f'Unsafe relative path: {relative}')
    path = root.joinpath(*posix.parts)
    reject_link_ancestors(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise DatasetError(f'Path escapes root: {relative}')
    return path


def inspect_image(path):
    if path.suffix.lower() not in EXTENSIONS:
        raise DatasetError(f'Unsupported image extension: {path}')
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        orientation = image.getexif().get(274, 1)
        if width <= 0 or height <= 0:
            raise DatasetError(f'Invalid image dimensions: {path}')
    return {'sha256': sha256(path), 'width': width, 'height': height,
            'exif_orientation': orientation, 'orientation_concern': orientation != 1}


def allocate_groups(records, seed):
    """Greedy absolute-deviation allocation, then strictly improving moves."""
    groups = {}
    pinned = set()
    for record in records:
        outlet = record['outlet_id']
        groups.setdefault(outlet, []).append(record)
        if record['benchmark_id']:
            pinned.add(outlet)
    targets = {split: len(records) * ratio for split, ratio in RATIOS.items()}
    counts = dict.fromkeys(SPLITS, 0)
    assigned = {}
    for outlet in sorted(pinned):
        assigned[outlet] = 'test'
        counts['test'] += len(groups[outlet])
    ordered = sorted(set(groups) - pinned)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda outlet: -len(groups[outlet]))

    def score(candidate):
        return sum(abs(candidate[s] - targets[s]) for s in SPLITS)

    for outlet in ordered:
        size = len(groups[outlet])
        destination = min(SPLITS, key=lambda s: score({**counts, s: counts[s] + size}))
        assigned[outlet] = destination
        counts[destination] += size
    while True:
        current = score(counts)
        best = None
        best_score = current
        for outlet in ordered:
            origin = assigned[outlet]
            size = len(groups[outlet])
            for destination in SPLITS:
                if destination == origin:
                    continue
                candidate = {**counts, origin: counts[origin] - size,
                             destination: counts[destination] + size}
                candidate_score = score(candidate)
                if candidate_score < best_score - 1e-9:
                    best = outlet, destination, candidate
                    best_score = candidate_score
        if best is None:
            break
        outlet, destination, counts = best
        assigned[outlet] = destination
    return assigned, counts, targets


def check_label(path, annotation_status):
    """Validate reviewed negatives explicitly and YOLO boxes in pixel axes."""
    if annotation_status not in ANNOTATION_STATUSES:
        raise DatasetError(f'Unknown annotation_status: {annotation_status}')
    if not path.exists():
        if annotation_status == 'reviewed_positive':
            raise DatasetError(f'Reviewed positive requires a nonempty label file: {path}')
        return 'missing'
    lines = [line for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not lines:
        if annotation_status != 'reviewed_negative':
            raise DatasetError(f'Empty label requires reviewed_negative: {path}')
        return 'reviewed_negative'
    if annotation_status == 'reviewed_negative':
        raise DatasetError(f'Reviewed negative has boxes: {path}')
    for number, line in enumerate(lines, 1):
        columns = line.split()
        if len(columns) != 5 or columns[0] != '0':
            raise DatasetError(f'Invalid YOLO row/class at {path}:{number}')
        try:
            x, y, w, h = map(float, columns[1:])
        except ValueError as exc:
            raise DatasetError(f'Non-numeric YOLO row at {path}:{number}') from exc
        if (not all(math.isfinite(v) for v in (x, y, w, h))
                or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                or x - w / 2 < -1e-6 or y - h / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6):
            raise DatasetError(f'YOLO box outside bounds at {path}:{number}')
    return 'reviewed_positive' if annotation_status == 'reviewed_positive' else 'pending'


def expected_image_path(record):
    extension = PurePosixPath(record['source_path']).suffix.lower()
    stem = record['benchmark_id'] or f"img_{record['sha256']}"
    return f"dataset/images/{record['split']}/{stem}{extension}"


def expected_label_path(record):
    return str(PurePosixPath(expected_image_path(record)).with_suffix('.txt')).replace('images/', 'labels/', 1)


def dataset_yaml(root):
    return {'path': str((root / 'dataset').resolve()), 'train': 'images/train', 'val': 'images/val',
            'test': 'images/test', 'names': {0: 'DENA'}}


def check_dataset_tree(root, expected_images, expected_labels):
    """Reject unexpected files and linked paths before any destination writes."""
    reject_link_ancestors(root)
    if root.exists() and not root.is_dir():
        raise DatasetError(f'Dataset root is not a directory: {root}')
    for path in (root / 'manifest.json', root / 'configs', root / 'configs' / 'data.yaml', root / 'dataset'):
        reject_link_ancestors(path)
    if (root / 'configs').exists() and not (root / 'configs').is_dir():
        raise DatasetError(f'Configs directory is a file: {root / "configs"}')
    dataset = root / 'dataset'
    if dataset.exists():
        if not dataset.is_dir():
            raise DatasetError(f'Dataset is not a directory: {dataset}')
        for child in dataset.iterdir():
            if is_link(child) or child.name not in {'images', 'labels'}:
                raise DatasetError(f'Unexpected or unsafe dataset entry: {child}')
    for category, expected in [('images', expected_images), ('labels', expected_labels)]:
        parent = dataset / category
        if parent.exists():
            if not parent.is_dir():
                raise DatasetError(f'Dataset directory is a file: {parent}')
            for path in parent.rglob('*'):
                if is_link(path):
                    raise DatasetError(f'Unsafe dataset symlink/junction: {path}')
                relative = path.relative_to(root).as_posix()
                if path.is_dir():
                    if relative not in {f'dataset/{category}/{s}' for s in SPLITS}:
                        raise DatasetError(f'Unexpected dataset directory: {relative}')
                elif relative not in expected:
                    raise DatasetError(f'Unexpected dataset file: {relative}')
