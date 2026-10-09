#!/usr/bin/env python3
"""Inventory, split, and validate DENA data without manufacturing annotations."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

from dataset_utils import (
    ANNOTATION_STATUSES, BENCHMARKS, DatasetError, EXTENSIONS, RATIOS, SPLITS,
    allocate_groups, check_dataset_tree, check_label, contained_path, dataset_yaml,
    expected_image_path, expected_label_path, inferred_outlet, inspect_image,
    reject_link_ancestors,
)


def source_root(value):
    root = Path(value).absolute()
    reject_link_ancestors(root)
    if not root.is_dir():
        raise DatasetError(f'Source root does not exist: {root}')
    return root.resolve()


def inventory(args):
    root = source_root(args.source)
    output = Path(args.output).absolute()
    reject_link_ancestors(output)
    if output.resolve().is_relative_to(root):
        raise DatasetError('Inventory output must be outside the source archive')
    records = []
    errors = 0
    for path in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        if path.suffix.lower() not in EXTENSIONS or path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        record = {'source_path': relative, 'outlet_id': inferred_outlet(relative),
                  'benchmark_id': BENCHMARKS.get(relative, ''), 'annotation_status': 'pending',
                  'status': 'ok', 'error': ''}
        try:
            record.update(inspect_image(contained_path(root, relative)))
        except (OSError, ValueError, SyntaxError) as exc:
            record.update(status='error', error=str(exc))
            errors += 1
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['source_path', 'outlet_id', 'benchmark_id', 'annotation_status', 'status',
              'error', 'sha256', 'width', 'height', 'exif_orientation', 'orientation_concern']
    with output.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    orientation = sum(bool(r.get('orientation_concern')) for r in records)
    print(f'Inventory: {len(records)} images, {errors} errors, {orientation} EXIF orientation concerns. {output}')
    return 1 if errors else 0


def read_selection(root, selected):
    records = []
    seen_paths = set()
    seen_content = set()
    found_benchmarks = set()
    with selected.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {'source_path', 'outlet_id'}.issubset(reader.fieldnames):
            raise DatasetError('Selection CSV requires source_path and outlet_id columns')
        for number, row in enumerate(reader, 2):
            relative = (row.get('source_path') or '').strip()
            outlet = (row.get('outlet_id') or '').strip()
            benchmark = (row.get('benchmark_id') or '').strip()
            if relative in seen_paths:
                raise DatasetError(f'Duplicate selection: {relative}')
            path = contained_path(root, relative)
            if not path.is_file():
                raise DatasetError(f'Missing selected image: {relative}')
            if not outlet or not outlet.isascii() or not outlet.isdigit():
                raise DatasetError(f'Missing or invalid numeric outlet_id at CSV row {number}')
            inferred = inferred_outlet(relative)
            if inferred and inferred != outlet:
                raise DatasetError(f'outlet_id disagrees with filename: {relative}: {outlet} != {inferred}')
            expected_benchmark = BENCHMARKS.get(relative, '')
            if benchmark and benchmark != expected_benchmark:
                raise DatasetError(f'Wrong/unknown benchmark_id for {relative}: expected {expected_benchmark!r}')
            benchmark = expected_benchmark
            if benchmark:
                if benchmark in found_benchmarks:
                    raise DatasetError(f'Duplicate benchmark id: {benchmark}')
                found_benchmarks.add(benchmark)
            metadata = inspect_image(path)
            if metadata['sha256'] in seen_content:
                raise DatasetError(f'Duplicate selected image content: {relative}')
            seen_paths.add(relative)
            seen_content.add(metadata['sha256'])
            records.append({'source_path': relative, 'outlet_id': outlet, 'benchmark_id': benchmark,
                            'annotation_status': 'pending', **metadata})
    missing = set(BENCHMARKS.values()) - found_benchmarks
    if missing:
        raise DatasetError(f'Missing required benchmarks: {", ".join(sorted(missing))}')
    return sorted(records, key=lambda record: record['source_path'])


def make_manifest(root, records, seed):
    assignment, counts, targets = allocate_groups(records, seed)
    for record in records:
        record['split'] = assignment[record['outlet_id']]
        record['image_path'] = expected_image_path(record)
        record['label_path'] = expected_label_path(record)
    data = {'version': 1, 'seed': seed, 'source_root': str(root), 'ratios': RATIOS,
            'counts': counts, 'images': records}
    print('Split: ' + ', '.join(f'{split}={counts[split]} (target {targets[split]:.2f}, deviation {counts[split] - targets[split]:+.2f})' for split in SPLITS))
    print(f'Total absolute deviation: {sum(abs(counts[s] - targets[s]) for s in SPLITS):.2f}')
    orientation = sum(record['orientation_concern'] for record in records)
    if orientation:
        print(f'EXIF orientation concerns: {orientation}; files preserved byte-for-byte. Review before labeling.')
    return data


def load_manifest(path):
    with path.open(encoding='utf-8-sig') as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('images'), list):
        raise DatasetError('Unsupported or invalid manifest (expected version 1)')
    return data


def immutable_manifest(data):
    return {**data, 'images': [{k: v for k, v in record.items() if k != 'annotation_status'} for record in data['images']]}


def preflight(output, data):
    images = {r['image_path'] for r in data['images']}
    labels = {r['label_path'] for r in data['images']}
    if len(images) != len(data['images']) or len(labels) != len(data['images']):
        raise DatasetError('Output filename collision')
    check_dataset_tree(output, images, labels)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = load_manifest(manifest_path)
        if immutable_manifest(old) != immutable_manifest(data):
            raise DatasetError('Existing manifest conflicts with the requested selection/split')
        for record, previous in zip(data['images'], old['images']):
            status = previous.get('annotation_status')
            if status not in ANNOTATION_STATUSES:
                raise DatasetError(f'Unknown existing annotation status: {status}')
            record['annotation_status'] = status
    config_path = output / 'configs' / 'data.yaml'
    if config_path.exists():
        with config_path.open(encoding='utf-8-sig') as handle:
            config = yaml.safe_load(handle)
        if config != dataset_yaml(output):
            raise DatasetError('Existing data.yaml conflicts with dataset root or class/split contract')
    for record in data['images']:
        path = contained_path(output, record['image_path'])
        if path.exists() and inspect_image(path) != {key: record[key] for key in ('sha256', 'width', 'height', 'exif_orientation', 'orientation_concern')}:
            raise DatasetError(f'Existing image conflicts with selection: {record["image_path"]}')
        label = contained_path(output, record['label_path'])
        if label.exists():
            if not manifest_path.exists():
                raise DatasetError(f'Existing label has no manifest provenance: {record["label_path"]}')
        if manifest_path.exists():
            check_label(label, record['annotation_status'])


def publish(root, output, data):
    """Stage all copies before publish; existing datasets publish metadata last."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.staging-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        for category in ('images', 'labels'):
            for split_name in SPLITS:
                (staged / 'dataset' / category / split_name).mkdir(parents=True)
        for record in data['images']:
            source = contained_path(root, record['source_path'])
            target = staged / record['image_path']
            shutil.copy2(source, target)
            if inspect_image(target) != {key: record[key] for key in ('sha256', 'width', 'height', 'exif_orientation', 'orientation_concern')}:
                raise DatasetError(f'Source changed during copy: {record["source_path"]}')
        (staged / 'configs').mkdir()
        (staged / 'configs' / 'data.yaml').write_text(yaml.safe_dump(dataset_yaml(output), sort_keys=False, allow_unicode=True), encoding='utf-8')
        (staged / 'manifest.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        # Recheck after staging to avoid overwriting changes made during long copies.
        preflight(output, data)
        (staged / 'manifest.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        if not output.exists():
            os.replace(staged, output)
            return
        for category in ('images', 'labels'):
            for split_name in SPLITS:
                (output / 'dataset' / category / split_name).mkdir(parents=True, exist_ok=True)
        for record in data['images']:
            target = output / record['image_path']
            if not target.exists():
                os.replace(staged / record['image_path'], target)
        (output / 'configs').mkdir(exist_ok=True)
        for filename in ('configs/data.yaml', 'manifest.json'):
            os.replace(staged / filename, output / filename)


def split(args):
    root = source_root(args.source)
    selected = Path(args.selection)
    output = Path(args.output).absolute()
    reject_link_ancestors(output)
    output = output.resolve()
    if output == root or root.is_relative_to(output) or output.is_relative_to(root):
        raise DatasetError('Source and dataset output must be separate, non-overlapping directories')
    records = read_selection(root, selected)
    data = make_manifest(root, records, args.seed)
    preflight(output, data)
    if args.dry_run:
        print('Dry run: no files or directories written.')
        return 0
    publish(root, output, data)
    print(f'Dataset ready for annotation: {output}. No placeholder labels created.')
    return 0


def validate_manifest_contract(data, root):
    if data.get('ratios') != RATIOS or not isinstance(data.get('seed'), int):
        raise DatasetError('Invalid manifest ratios or seed')
    raw_root = Path(data.get('source_root', ''))
    if not raw_root.is_absolute():
        raise DatasetError('Manifest source_root must be absolute')
    counts = dict.fromkeys(SPLITS, 0)
    seen_paths, seen_content, seen_images, seen_labels = set(), set(), set(), set()
    found_benchmarks = set()
    outlet_splits = {}
    for record in data['images']:
        required = {'source_path', 'outlet_id', 'benchmark_id', 'annotation_status', 'sha256',
                    'width', 'height', 'split', 'image_path', 'label_path'}
        if not isinstance(record, dict) or not required.issubset(record):
            raise DatasetError('Manifest image record is missing required fields')
        relative = record['source_path']
        # Canonical path syntax and containment are enforced without needing the raw archive online.
        contained_path(root, relative)
        outlet = record['outlet_id']
        if not isinstance(outlet, str) or not outlet or not outlet.isascii() or not outlet.isdigit():
            raise DatasetError('Invalid manifest outlet_id')
        inferred = inferred_outlet(relative)
        if inferred and outlet != inferred:
            raise DatasetError(f'outlet_id disagrees with manifest source filename: {relative}')
        expected_benchmark = BENCHMARKS.get(relative, '')
        if record['benchmark_id'] != expected_benchmark:
            raise DatasetError(f'Invalid benchmark mapping: {relative}')
        if expected_benchmark:
            if expected_benchmark in found_benchmarks or record['split'] != 'test':
                raise DatasetError(f'Benchmark must occur once in test: {expected_benchmark}')
            found_benchmarks.add(expected_benchmark)
        split_name = record['split']
        if split_name not in SPLITS:
            raise DatasetError(f'Unknown split: {split_name}')
        if outlet in outlet_splits and outlet_splits[outlet] != split_name:
            raise DatasetError(f'Outlet leakage: {outlet}')
        outlet_splits[outlet] = split_name
        if (not isinstance(record['sha256'], str) or len(record['sha256']) != 64
                or any(ch not in '0123456789abcdef' for ch in record['sha256'])):
            raise DatasetError('Invalid manifest image hash')
        if (not isinstance(record['width'], int) or not isinstance(record['height'], int)
                or record['width'] <= 0 or record['height'] <= 0):
            raise DatasetError('Invalid manifest dimensions')
        if record['annotation_status'] not in ANNOTATION_STATUSES:
            raise DatasetError('Invalid annotation_status')
        if record['image_path'] != expected_image_path(record) or record['label_path'] != expected_label_path(record):
            raise DatasetError(f'Incorrect output path for {relative}')
        for field, seen in [('source_path', seen_paths), ('sha256', seen_content),
                            ('image_path', seen_images), ('label_path', seen_labels)]:
            if record[field] in seen:
                raise DatasetError(f'Duplicate manifest {field}: {record[field]}')
            seen.add(record[field])
        counts[split_name] += 1
    if found_benchmarks != set(BENCHMARKS.values()):
        raise DatasetError('Manifest missing required exact benchmark source paths')
    if data.get('counts') != counts:
        raise DatasetError('Manifest counts disagree with image records')
    return seen_images, seen_labels


def validate(args):
    root = Path(args.root).absolute()
    reject_link_ancestors(root)
    root = root.resolve()
    data = load_manifest(root / 'manifest.json')
    images, labels = validate_manifest_contract(data, root)
    check_dataset_tree(root, images, labels)
    for category in ('images', 'labels'):
        for split_name in SPLITS:
            if not (root / 'dataset' / category / split_name).is_dir():
                raise DatasetError(f'Missing dataset directory: dataset/{category}/{split_name}')
    with (root / 'configs' / 'data.yaml').open(encoding='utf-8-sig') as handle:
        config = yaml.safe_load(handle)
    if config != dataset_yaml(root):
        raise DatasetError('data.yaml does not match dataset root, splits, or DENA class')
    errors = []
    missing = pending = reviewed = 0
    for record in data['images']:
        try:
            info = inspect_image(contained_path(root, record['image_path']))
            for field in ('sha256', 'width', 'height'):
                if record[field] != info[field]:
                    raise DatasetError(f'Image {field} mismatch: {record["image_path"]}')
            label_state = check_label(contained_path(root, record['label_path']), record['annotation_status'])
            if label_state == 'missing':
                missing += 1
            elif label_state == 'pending':
                pending += 1
            else:
                reviewed += 1
        except (OSError, ValueError, SyntaxError) as exc:
            errors.append(str(exc))
    print(f'Validation: {len(data["images"])} images, {len(errors)} structural errors, {missing} missing labels, {pending} pending labels, {reviewed} reviewed annotations.')
    print(f'Annotation readiness: {missing + pending} pending review or missing annotations.')
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    if errors:
        return 1
    if args.require_annotations and (missing or pending):
        return 2
    return 0


def main(argv=None):
    # Windows redirected streams otherwise use the active ANSI code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inventory_parser = commands.add_parser('inventory', help='Inspect JPG/JPEG/PNG files recursively')
    inventory_parser.add_argument('--source', required=True)
    inventory_parser.add_argument('--output', required=True)
    inventory_parser.set_defaults(run=inventory)
    split_parser = commands.add_parser('split', help='Copy an outlet-grouped, benchmark-pinned selection')
    split_parser.add_argument('--source', required=True)
    split_parser.add_argument('--selection', required=True)
    split_parser.add_argument('--output', required=True)
    split_parser.add_argument('--seed', type=int, default=42)
    split_parser.add_argument('--dry-run', action='store_true')
    split_parser.set_defaults(run=split)
    validate_parser = commands.add_parser('validate', help='Check dataset structure and annotation readiness')
    validate_parser.add_argument('--root', required=True)
    validate_parser.add_argument('--require-annotations', action='store_true')
    validate_parser.set_defaults(run=validate)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, SyntaxError, yaml.YAMLError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
