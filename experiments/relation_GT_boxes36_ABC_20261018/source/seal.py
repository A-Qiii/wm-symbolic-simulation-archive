"""Verify existing results and seal this bounded 36-root experiment. No fitting."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import shutil
import os

W = Path('/home/robot/Documents/Codex/2026-10-13/task-4')
P = Path('/home/robot/research/projects/wm_symbolic')
E = P / 'experiments/relation_GT_boxes36_ABC_20261018'
R = P / 'reports/relation_GT_boxes36_ABC_20261018'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(p, x):
    assert not Path(p).exists(), p
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def check_map(mapping):
    for p, expected in mapping.items():
        assert sha(p) == expected, p
    return len(mapping)

assert not (E / 'configs/delivery_manifest.json').exists()
refs = read(E / 'configs/input_references.json')
old = []
for ref in refs['old_manifests']:
    assert sha(ref['path']) == ref['sha256'], ref['path']
    m = read(ref['path'])
    count = check_map(m['sha256'])
    assert count == m['files'] == ref['files_verified']
    assert all(not (Path(p).stat().st_mode & 0o222) for p in m['sha256'])
    old.append(dict(ref, files_reverified=count, read_only=True))
inputs_verified = check_map(refs['sha256'])
frozen = {}
for name in ('precollection_frozen', 'data_frozen', 'models_frozen'):
    frozen[name] = check_map(read(E / f'configs/{name}.json')['sha256'])
models = read(E / 'configs/models_frozen.json')
assert sha(E / 'configs/train_normalization.npz') == models['normalization_sha256']
assert sha(E / 'configs/protocol.json') == sha(R / 'FROZEN_PROTOCOL.json')
guide = P / 'source/世界模型与TS-LTL符号规划_核心预实验方案.md'
assert sha(guide) == 'f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4'
features = read(E / 'results/features.json')
for f in features['features']:
    assert sha(E / f"data/features/{f['split']}.npz") == f['sha256']
prediction = read(E / 'results/prediction.json')
assert len(prediction['passes']) == 12 and prediction['threshold'] == 0.5
for f in prediction['passes']:
    assert sha(E / f"results/predictions_{f['arm']}_seed{f['seed']}_{f['split']}.npz") == f['sha256']
protocol = read(E / 'configs/protocol.json')
roots = protocol['roots']
attempts = read(E / 'results/attempts.json')
assert len(roots) == len(attempts) == 36
assert len(set(x['root_id'] for x in roots)) == 36
assert [x['root_id'] for x in attempts] == [x['root_id'] for x in roots]
assert Counter(x['split'] for x in roots) == {'train': 24, 'dev': 12}
assert Counter(x['scene'] for x in roots) == {'A': 12, 'B': 12, 'C': 12}
assert [x['seed'] for x in roots] == list(range(261019500, 261019536))
training = read(E / 'results/training.json')['fits']
training_attempts = read(E / 'results/training_attempts.json')
assert len(training) == len(training_attempts) == 6
assert all(x['updates'] == 600 for x in training)
assert all(x['status'] == 'completed' for x in training_attempts)
assert {(x['arm'], x['seed']) for x in training} == {(a,s) for a in ('object','pair','geometry') for s in (0,1)}
assert len(list((E/'models').glob('*.pt'))) == 12
assert len(list((E/'results').glob('predictions_*.npz'))) == 12
gate = read(E / 'results/data_gate.json')
assert gate['passed'] and not gate['failures'] and gate['D_sample_reads'] == 0
collection = read(E / 'results/collection_summary.json')
assert (collection['roots'], collection['states'], collection['records']) == (36,264,1584)
assert len(list((E/'data/public').glob('*.npz'))) == 264
independent = read(E / 'results/independent_verification.json')
assert independent['metric_rows_recomputed'] == 360
assert independent['expanded_root_bootstrap_checks'] == 60
assert independent['geometry_feature_rows_recomputed'] == 1390
assert independent['additional_training_or_prediction_passes'] == 0
library = read(W / 'library_relation36_20261018/confirmed.json')
assert library['library_file_id'] == 'libfile_8407fe4b6c848191a3eb0f1730423969'
assert sha(library['upload_path']) == sha(R / 'figures/01_readout_comparison.png')
for name in library['xattrs_applied']:
    assert os.getxattr(library['upload_path'], name)
write(R / 'LIBRARY.json', library)
shutil.copy2(__file__, E / 'source/seal.py')
for src in (W/'relation36_src').glob('*.py'):
    assert sha(src) == sha(E/'source'/src.name), src
verification = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'passed': True,
    'old_experiments': old,
    'input_references_verified': inputs_verified,
    'frozen_file_counts_verified': frozen,
    'guide_sha256': sha(guide),
    'protocol_sha256': sha(E/'configs/protocol.json'),
    'report_protocol_copy_identical': True,
    'feature_files_verified': 2,
    'prediction_files_verified': 12,
    'initial_and_final_model_file_count': 12,
    'model_fits': 6,
    'updates_per_fit': 600,
    'roots': 36, 'train_roots': 24, 'dev_roots': 12, 'states': 264,
    'data_gate_passed': True,
    'independent_metrics': independent,
    'D_samples_features_predictions_scores_opened_in_this_experiment': False,
    'historical_registry_metadata_only_used_for_exclusion': True,
    'figures_visually_reviewed': ['01_readout_comparison','02_color_camera','03_first_error_and_geometry_alias'],
    'library_upload_file_hash_and_returned_xattrs_verified': True,
    'additional_collection_training_prediction_in_finalization': 0,
    'full_world_model_guide_completed': False,
}
write(E / 'results/final_verification.json', verification)
files = sorted([p for base in (E,R) for p in base.rglob('*') if p.is_file()])
assert all(not p.is_symlink() for p in files)
manifest = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'scope': '36 independent A/B/C roots; conditional Holding readout; three frozen input comparisons, two seeds; no D/WM/closed loop',
    'outcome': 'gate passed; pair improves over object; geometry-only control stronger; no additional visual relation benefit established',
    'roots': 36, 'states': 264, 'fits': 6,
    'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
    'sha256': {str(p): sha(p) for p in files},
    'old32_and_pilot8_unchanged': True,
    'complete_guide_finished': False,
}
mp = E / 'configs/delivery_manifest.json'
write(mp, manifest)
for p in files + [mp]:
    p.chmod(0o444)
assert check_map(manifest['sha256']) == manifest['files']
assert all(not (p.stat().st_mode & 0o222) for p in files + [mp])
receipt = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'experiment': str(E), 'report': str(R/'REPORT.md'),
    'manifest': str(mp), 'manifest_sha256': sha(mp),
    'files': manifest['files'], 'bytes': manifest['bytes'],
    'read_only_verified': True, 'library_file_id': library['library_file_id'],
    'file_id': library['file_id'],
}
write(W / 'relation36_src/delivery_receipt.json', receipt)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
