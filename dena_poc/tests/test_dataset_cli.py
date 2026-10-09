import csv
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'setup_dataset.py'
BENCHMARKS = {
    'MR1 TOHIR/1л. DENA полкаси - тугирлаганимдан кейинги холат-5303-img-d6_1236-1790078345789.jpg': '5303',
    'SR6 Ibrohim/DENA Холодильник -4647-img-c7_5013-1790068813143.jpg': '4647',
    'SR6 Ibrohim/DENA LIDER точкада-XML_IDimg-e3_5140-1790064417011.jpg': '5140',
    'CH-5 Зилола/1л. DENA полкаси - тугирлаганимдан кейинги холат-XML_IDimg-e8_9615-1790074129177.jpg': '9615',
    'CH-3 Muhayyo/1л. DENA полкаси - тугирлаганимдан кейинги холат-XML_IDimg-f1_7663-1790066200804.jpg': '7663',
}


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, encoding='utf-8')


def picture(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (16 + n, 20 + n), (n * 17 % 255, n * 31 % 255, n * 7 % 255)).save(path)


def selection(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source_path', 'outlet_id', 'benchmark_id', 'review_status'])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'raw Unicode полка'
    rows = []
    for n, (name, benchmark) in enumerate(BENCHMARKS.items(), 1):
        picture(root / name, n)
        outlet = re.search(r'img-[a-z0-9]+_(\d+)-', name).group(1)
        rows.append(dict(source_path=name, outlet_id=outlet, benchmark_id=benchmark, review_status='selected'))
    for n in range(6, 36):
        outlet = '1236' if n == 6 else str(10000 + (n - 7) // 3)
        name = f'Rep {n % 2}/Shelf-img-aa_{outlet}-{n}.png'
        picture(root / name, n)
        rows.append(dict(source_path=name, outlet_id=outlet, benchmark_id='', review_status='selected'))
    selected = tmp_path / 'selection.csv'
    selection(selected, rows)
    return root, selected, rows


def split(source, output, *extra):
    root, selected, _ = source
    return cli('split', '--source', root, '--selection', selected, '--output', output, '--seed', '42', *extra)


def manifest(output):
    return json.loads((output / 'manifest.json').read_text(encoding='utf-8'))


def test_inventory_reports_every_image_and_corruption(source, tmp_path):
    root, _, rows = source
    (root / 'broken.JPEG').write_bytes(b'not an image')
    output = tmp_path / 'inventory.csv'
    result = cli('inventory', '--source', root, '--output', output)
    assert result.returncode == 1, result.stderr
    with output.open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    assert len(records) == len(rows) + 1
    broken = next(x for x in records if x['source_path'] == 'broken.JPEG')
    assert broken['status'] == 'error' and broken['error']
    known = next(x for x in records if x['source_path'] in BENCHMARKS)
    assert known['outlet_id'] == '9615' or known['outlet_id'] in {'1236', '5013', '5140', '7663'}
    assert known['benchmark_id'] == BENCHMARKS[known['source_path']]
    assert known['annotation_status'] == 'pending'
    assert int(known['width']) > 0 and len(known['sha256']) == 64


def test_split_is_grouped_deterministic_and_raw_preserved(source, tmp_path):
    root, selected, rows = source
    original = {r['source_path']: hashlib.sha256((root / r['source_path']).read_bytes()).hexdigest() for r in rows}
    output = tmp_path / 'dataset'
    result = split(source, output)
    assert result.returncode == 0, result.stderr
    data = manifest(output)
    assert data['version'] == 1 and data['seed'] == 42
    assert len(data['images']) == len(rows)
    groups = {}
    for record in data['images']:
        groups.setdefault(record['outlet_id'], set()).add(record['split'])
        assert record['annotation_status'] == 'pending'
        assert not (output / record['label_path']).exists()
        assert hashlib.sha256((output / record['image_path']).read_bytes()).hexdigest() == original[record['source_path']]
        if record['benchmark_id']:
            assert record['split'] == 'test'
            assert Path(record['image_path']).stem == record['benchmark_id']
    assert all(len(splits) == 1 for splits in groups.values())
    assert groups['1236'] == {'test'}
    config = yaml.safe_load((output / 'configs' / 'data.yaml').read_text(encoding='utf-8'))
    assert config['path'] == str((output / 'dataset').resolve()) and config['names'] == {0: 'DENA'}
    selection(selected, list(reversed(rows)))
    second = tmp_path / 'second'
    assert split(source, second).returncode == 0
    assert data['images'] == manifest(second)['images']
    assert original == {r['source_path']: hashlib.sha256((root / r['source_path']).read_bytes()).hexdigest() for r in rows}


def test_dry_run_does_not_create_output(source, tmp_path):
    output = tmp_path / 'absent' / 'dataset'
    result = split(source, output, '--dry-run')
    assert result.returncode == 0, result.stderr
    assert not output.parent.exists()
    assert 'deviation' in result.stdout.lower()


@pytest.mark.parametrize('omit_column', [True, False])
def test_benchmark_tags_are_inferred_when_not_supplied(source, tmp_path, omit_column):
    _, selected, rows = source
    if omit_column:
        with selected.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['source_path', 'outlet_id'])
            writer.writeheader()
            writer.writerows({k: row[k] for k in ['source_path', 'outlet_id']} for row in rows)
    else:
        selection(selected, [{**row, 'benchmark_id': ''} for row in rows])
    output = tmp_path / 'dataset'
    result = split(source, output)
    assert result.returncode == 0, result.stderr
    assert {r['benchmark_id'] for r in manifest(output)['images'] if r['benchmark_id']} == set(BENCHMARKS.values())


def test_existing_project_files_are_preserved(source, tmp_path):
    output = tmp_path / 'project'
    (output / 'configs').mkdir(parents=True)
    (output / 'scripts').mkdir()
    selection_file = output / 'configs' / 'selection.csv'
    selection_file.write_text('user selection')
    (output / 'configs' / 'baseline.yaml').write_text('train parameters')
    script_file = output / 'scripts' / 'train.py'
    script_file.write_text('user script')
    assert split(source, output).returncode == 0
    assert selection_file.read_text() == 'user selection'
    assert script_file.read_text() == 'user script'


@pytest.mark.parametrize('mutation', ['missing_benchmark', 'wrong_benchmark', 'unknown_benchmark', 'duplicate_path', 'duplicate_content', 'escape', 'absolute', 'missing', 'bad_outlet', 'missing_outlet', 'corrupt', 'benchmark_impostor'])
def test_bad_selection_rejected_without_output(source, tmp_path, mutation):
    root, selected, rows = source
    rows = [dict(row) for row in rows]
    if mutation == 'missing_benchmark': rows.pop(0)
    elif mutation == 'wrong_benchmark': rows[0]['benchmark_id'] = '4647'
    elif mutation == 'unknown_benchmark': rows[-1]['benchmark_id'] = 'benchmark_99'
    elif mutation == 'duplicate_path': rows.append(rows[-1])
    elif mutation == 'duplicate_content': (root / rows[-1]['source_path']).write_bytes((root / rows[-2]['source_path']).read_bytes())
    elif mutation == 'escape': rows[-1]['source_path'] = '../outside.png'
    elif mutation == 'absolute': rows[-1]['source_path'] = str((root / rows[-1]['source_path']).resolve())
    elif mutation == 'missing': rows[-1]['source_path'] = 'not-found.png'
    elif mutation == 'bad_outlet': rows[-1]['outlet_id'] = 'wrong'
    elif mutation == 'missing_outlet': rows[-1]['outlet_id'] = ''
    elif mutation == 'corrupt': (root / rows[-1]['source_path']).write_bytes(b'broken')
    elif mutation == 'benchmark_impostor':
        name = 'Different Rep/' + Path(rows[0]['source_path']).name
        (root / name).parent.mkdir()
        (root / rows[0]['source_path']).rename(root / name)
        rows[0]['source_path'] = name
    selection(selected, rows)
    output = tmp_path / 'output'
    result = split(source, output)
    assert result.returncode == 1, result.stdout
    assert not output.exists()


def test_rerun_preserves_reviewed_annotations_and_rejects_conflicts(source, tmp_path):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    data = manifest(output)
    record = data['images'][0]
    record['annotation_status'] = 'reviewed_positive'
    label = output / record['label_path']
    label.write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert split(source, output).returncode == 0
    assert manifest(output)['images'][0]['annotation_status'] == 'reviewed_positive'
    assert label.read_text() == '0 0.5 0.5 0.2 0.2\n'
    conflict = output / 'dataset' / 'images' / 'train' / 'stale.jpg'
    conflict.write_bytes(b'stale')
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert split(source, output).returncode == 1
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_conflicting_yaml_and_unplanned_label_refused_before_writes(source, tmp_path):
    output = tmp_path / 'dataset'
    output.mkdir()
    (output / 'configs').mkdir()
    (output / 'configs' / 'data.yaml').write_text('names: {0: OTHER}\n')
    assert split(source, output).returncode == 1
    assert not (output / 'manifest.json').exists()
    (output / 'configs' / 'data.yaml').unlink()
    (output / 'dataset' / 'labels' / 'train').mkdir(parents=True)
    (output / 'dataset' / 'labels' / 'train' / 'unknown.txt').write_text('')
    assert split(source, output).returncode == 1
    assert not (output / 'manifest.json').exists()


def test_configs_file_conflict_is_rejected_before_dataset_writes(source, tmp_path):
    output = tmp_path / 'project'
    output.mkdir()
    (output / 'configs').write_text('conflict')
    assert split(source, output).returncode == 1
    assert not (output / 'dataset').exists()
    assert (output / 'configs').read_text() == 'conflict'


def test_inventory_detects_exif_and_preserves_orientation(source, tmp_path):
    root, _, _ = source
    path = root / 'Rep/img-ab_99999-1.JPG'
    path.parent.mkdir()
    image = Image.new('RGB', (40, 30), 'blue')
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    original = path.read_bytes()
    output = tmp_path / 'inventory.csv'
    assert cli('inventory', '--source', root, '--output', output).returncode == 0
    with output.open(encoding='utf-8-sig', newline='') as handle:
        record = next(row for row in csv.DictReader(handle) if row['source_path'] == 'Rep/img-ab_99999-1.JPG')
    assert record['exif_orientation'] == '6'
    assert record['orientation_concern'] == 'True'
    assert record['width'] == '40' and record['height'] == '30'
    assert path.read_bytes() == original


def test_inventory_cannot_overwrite_source_archive(source):
    root, _, rows = source
    target = root / rows[0]['source_path']
    original = target.read_bytes()
    result = cli('inventory', '--source', root, '--output', target)
    assert result.returncode == 1
    assert target.read_bytes() == original


def test_pending_nonempty_box_requires_actual_review(source, tmp_path):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    data = manifest(output)
    for record in data['images']:
        (output / record['label_path']).write_text('0 0.5 0.5 0.2 0.2\n')
    assert cli('validate', '--root', output).returncode == 0
    assert cli('validate', '--root', output, '--require-annotations').returncode == 2
    for record in data['images']:
        record['annotation_status'] = 'reviewed_positive'
    (output / 'manifest.json').write_text(json.dumps(data), encoding='utf-8')
    assert cli('validate', '--root', output, '--require-annotations').returncode == 0


@pytest.mark.parametrize('status,box', [('reviewed_negative', '0 0.5 0.5 0.2 0.2'), ('reviewed_positive', '')])
def test_review_status_must_agree_with_annotation_contents(source, tmp_path, status, box):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    data = manifest(output)
    data['images'][0]['annotation_status'] = status
    (output / data['images'][0]['label_path']).write_text(box)
    (output / 'manifest.json').write_text(json.dumps(data), encoding='utf-8')
    assert cli('validate', '--root', output).returncode == 1


def test_reviewed_positive_requires_a_nonempty_annotation_file(source, tmp_path):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    data = manifest(output)
    data['images'][0]['annotation_status'] = 'reviewed_positive'
    (output / 'manifest.json').write_text(json.dumps(data), encoding='utf-8')
    assert cli('validate', '--root', output).returncode == 1


def test_existing_image_conflict_does_not_publish_other_files(source, tmp_path):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    data = manifest(output)
    (output / data['images'][0]['image_path']).write_bytes(b'conflict')
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert split(source, output).returncode == 1
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_validation_separates_pending_from_structural_errors(source, tmp_path):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    result = cli('validate', '--root', output)
    assert result.returncode == 0, result.stderr
    assert 'pending' in result.stdout.lower() and 'missing' in result.stdout.lower()
    assert cli('validate', '--root', output, '--require-annotations').returncode == 2
    data = manifest(output)
    for record in data['images']:
        record['annotation_status'] = 'reviewed_negative'
        (output / record['label_path']).write_text('', encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(data), encoding='utf-8')
    assert cli('validate', '--root', output, '--require-annotations').returncode == 0


@pytest.mark.parametrize('box', ['', '1 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2', '0 0.5 0.5 0 0.2', '0 0.95 0.5 0.2 0.2', '0 0.5 0.5 0.2', '0 0.5 0.5 1.1 0.2'])
def test_invalid_label_rows_fail(source, tmp_path, box):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    record = manifest(output)['images'][0]
    (output / record['label_path']).write_text(box, encoding='utf-8')
    assert cli('validate', '--root', output).returncode == 1


@pytest.mark.parametrize('damage', ['image_hash', 'dimensions', 'outlet_leakage', 'benchmark', 'unexpected_image', 'unexpected_label', 'yaml'])
def test_validator_detects_dataset_tampering(source, tmp_path, damage):
    output = tmp_path / 'dataset'
    assert split(source, output).returncode == 0
    data = manifest(output)
    if damage == 'image_hash': (output / data['images'][0]['image_path']).write_bytes(b'bad')
    elif damage == 'dimensions': data['images'][0]['width'] += 1
    elif damage == 'outlet_leakage': data['images'][-1]['outlet_id'] = data['images'][0]['outlet_id']
    elif damage == 'benchmark': data['images'][0]['benchmark_id'] = 'benchmark_99'
    elif damage == 'unexpected_image': picture(output / 'dataset' / 'images' / 'val' / 'extra.png', 2)
    elif damage == 'unexpected_label': (output / 'dataset' / 'labels' / 'test' / 'extra.txt').write_text('')
    elif damage == 'yaml': (output / 'configs' / 'data.yaml').write_text('path: wrong\n')
    (output / 'manifest.json').write_text(json.dumps(data), encoding='utf-8')
    assert cli('validate', '--root', output).returncode == 1


def test_unsafe_source_symlink_rejected(source, tmp_path):
    root, selected, rows = source
    outside = tmp_path / 'outside.png'
    picture(outside, 55)
    link = root / 'symlink.png'
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('Symlink creation unavailable on this Windows account')
    rows.append(dict(source_path='symlink.png', outlet_id='5500', benchmark_id='', review_status='selected'))
    selection(selected, rows)
    assert split(source, tmp_path / 'dataset').returncode == 1


def test_destination_symlink_rejected(source, tmp_path):
    output = tmp_path / 'dataset'
    output.mkdir()
    (output / 'dataset').mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    try:
        (output / 'dataset' / 'images').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation unavailable on this Windows account')
    assert split(source, output).returncode == 1
    assert not list(outside.iterdir())
