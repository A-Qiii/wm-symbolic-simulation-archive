"""Authorized frozen-model evaluation supplement; never modifies sealed artifacts."""
import os
os.environ['RSSM_SCALE_ROOTS'] = '40'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import sys, json, hashlib, datetime, time, csv, signal, fcntl, re, traceback
import numpy as np

P = Path('/home/robot/research/projects/wm_symbolic')
B = P / 'experiments/rssm_scaling_run_20261019'
S = P / 'experiments/rssm_h2_supplement_20261019'
os.environ['MPLCONFIGDIR'] = str(S / 'cache/matplotlib')
os.environ['TRITON_CACHE_DIR'] = str(S / 'cache/triton')
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = str(S / 'cache/torch_kernels')
sys.path.insert(0, str(B / 'source'))
import torch
import common
from fixed_dataset import CoreEpisodes
from wm_symbolic.world_model import WorldModel

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def verify_seal():
    path = B / 'configs/delivery_manifest.json'
    m = json.loads(path.read_text())
    assert len(m['sha256']) == 319
    for name, expected in m['sha256'].items():
        assert sha(name) == expected, 'SEALED FILE MISMATCH: ' + name
    return {'files': len(m['sha256']), 'manifest_sha256': sha(path), 'all_match': True}

def protections(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode = args[1]
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(c in mode for c in 'wa+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if writing and not p.is_relative_to(S) and p != Path('/dev/null'):
        raise RuntimeError('Write outside new supplement forbidden: ' + str(p))
    if p.is_relative_to(P / 'data/core1200_v1'):
        match = re.search(r'([ABCD]_core1200_[0-9]{2})', str(p))
        if match and match.group(1) not in ALLOWED:
            raise RuntimeError('Sample outside fixed test/C scope: ' + str(p))

def same_values(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind in 'fc':
        return np.array_equal(x, y, equal_nan=True)
    return np.array_equal(x, y)

def compare_previous(new_path, old_path):
    with np.load(new_path, allow_pickle=False) as z:
        new = dict(z)
    with np.load(old_path, allow_pickle=False) as z:
        old = dict(z)
    assert set(new) == set(old)
    mask = ~np.isin(new['mode'], ['free2', 'root_persistence2'])
    checks = {k: same_values(new[k][mask], old[k]) for k in old}
    delta = np.abs(new['prob'][mask] - old['prob'])
    result = {
        'new': str(new_path), 'original': str(old_path),
        'original_sha256': sha(old_path), 'new_sha256': sha(new_path),
        'existing_rows': len(old['prob']), 'additional_rows': int((~mask).sum()),
        'all_arrays_exactly_equal_including_NaN': all(checks.values()),
        'array_checks': checks, 'max_abs_probability_difference': float(delta.max()),
    }
    print('PARITY', json.dumps(result, ensure_ascii=False), flush=True)
    return result

def score_rollout(path, seed, split):
    with np.load(path, allow_pickle=False) as z:
        a = dict(z)
    selected = ((a['mode'] == 'current') & (a['step'] == 0)) | np.isin(a['mode'], ['free1', 'free2', 'free3', 'free4'])
    out = {k: v[selected] for k, v in a.items()}
    np.savez_compressed(S / 'data' / f'rollout_h0_h4_joint_seed{seed}_{split}.npz', **out)
    cohort = None
    metrics = []
    for h in range(5):
        mask = (out['step'] == h)
        p, y, roots, ids = (out[k][mask] for k in ['prob', 'cat', 'root_id', 'episode_id'])
        assert np.isfinite(p).all() and out['cat_valid'][mask].all()
        if cohort is None:
            cohort = list(ids)
        assert list(ids) == cohort and len(set(ids)) == len(ids)
        assert set(roots) == set(MEMBERS['test10' if split == 'test' else 'C20'])
        assert all(int((roots == r).sum()) == 3 for r in set(roots))
        pred = np.stack([p[:, lo:hi].argmax(-1) for lo, hi in [(0,8),(8,16),(16,24),(24,27),(27,30)]], 1)
        exact4 = (pred[:, :4] == y[:, :4]).all(1)
        exact5 = (pred == y).all(1)
        rel = ((p[:, [6,14,22]] >= .5) == (y[:, :3] == 6)).all(1) & ((p[:, [3,11,19]] >= .5) == (y[:, :3] == 3)).all(1) & (pred[:, 3] == y[:, 3])
        losses = np.stack([-np.log(np.clip(p[np.arange(len(p)), offset + y[:, j]], 1e-9, 1)) for j, offset in enumerate([0,8,16,24,27])], 1)
        macro = lambda values: float(np.mean([values[roots == r].mean() for r in sorted(set(roots))]))
        item = {'split': split, 'seed': seed, 'arm': 'joint', 'training_roots': 40, 'horizon': h, 'mode': 'current_root' if h == 0 else 'free_rollout', 'records': len(p), 'independent_roots': len(set(roots)), 'exact4_count': int(exact4.sum()), 'exact4_accuracy': macro(exact4), 'exact5_count': int(exact5.sum()), 'exact5_accuracy': macro(exact5), 'relation9_count': int(rel.sum()), 'relation9_accuracy': macro(rel), 'mean_NLL4': macro(losses[:, :4].mean(1)), 'mean_NLL5': macro(losses.mean(1))}
        metrics.append(item)
        if h == 2:
            print('H2_RESULT', json.dumps(item, ensure_ascii=False), flush=True)
    return metrics

def main():
    start = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Five-minute evaluation budget exceeded')))
    signal.alarm(300)
    assert not S.exists(), 'Do not overwrite an existing supplement'
    common.verify()
    before = verify_seal()
    assert torch.cuda.is_available(), 'Local GPU unavailable; no CPU substitution or environment change authorized'
    device_name = torch.cuda.get_device_name(0)
    lock = (P / 'runs/gpu.lock').open('r')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    S.mkdir()
    for name in ['source', 'data', 'results', 'cache']:
        (S / name).mkdir()
    sys.addaudithook(protections)
    save(S / 'results/preflight.json', {'utc': utc(), 'status': 'RUNNING', 'device': device_name, 'torch': torch.__version__, 'sealed_before': before, 'seeds': [0,1], 'splits': ['test','externalC'], 'posterior_samples': 8, 'torch_seed_each_assess': 81831, 'batch_episodes': 4, 'autocast': 'bfloat16', 'training_updates': 0, 'collection_episodes': 0, 'time_budget_seconds': 300})
    source_path = B / 'source/evaluate.py'
    original = source_path.read_text()
    needle = 'if t in [0,2,3]:'
    assert original.count(needle) == 1
    modified = original.replace(needle, 'if t in [0,1,2,3]:')
    (S / 'source/evaluate_with_h2.py').write_text(modified)
    (S / 'source/run.py').write_text(Path(__file__).read_text())
    scope = {'__name__': 'frozen_h2_supplement_eval', '__file__': str(S / 'source/evaluate_with_h2.py')}
    exec(compile(modified, scope['__file__'], 'exec'), scope)
    scope['E'] = S
    ordinary_probabilities = scope['probabilities']
    probability_calls = [0]
    def rng_checked_probabilities(*args):
        cpu_before, gpu_before = torch.get_rng_state(), torch.cuda.get_rng_state()
        result = ordinary_probabilities(*args)
        assert torch.equal(cpu_before, torch.get_rng_state()) and torch.equal(gpu_before, torch.cuda.get_rng_state()), 'Symbol probability extraction altered RNG state'
        probability_calls[0] += 1
        return result
    scope['probabilities'] = rng_checked_probabilities
    norm = json.loads((B / 'cells/roots40/configs/normalization.json').read_text())
    frozen = json.loads((B / 'configs/all_models_frozen.json').read_text())['sha256']
    metrics, parity, runs, model_hashes = [], [], [], {}
    for seed in [0,1]:
        model_path = B / f'cells/roots40/models/joint_seed{seed}/final6000.pt'
        assert sha(model_path) == frozen[str(model_path)]
        model_hashes[str(model_path)] = sha(model_path)
        m = WorldModel(skill_count=6, event_count=3).cuda()
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=True)
        m.load_state_dict(checkpoint['model'])
        m.eval().requires_grad_(False)
        state_before = common.th(m.state_dict())
        for split in ['test','externalC']:
            data = CoreEpisodes(P, split, normalizer=norm)
            runs.append(scope['assess'](m, data, 'joint', seed))
            name = f'predictions_joint_seed{seed}_{split}.npz'
            check = compare_previous(S / 'data' / name, B / 'cells/roots40/data' / name)
            parity.append(check)
            save(S / 'results/parity.json', parity)
            assert check['all_arrays_exactly_equal_including_NaN'], 'Mismatch: retain diagnostics and stop; never replace original report'
            for prefix in ['features']:
                name2 = f'{prefix}_joint_seed{seed}_{split}.npz'
                with np.load(S / 'data' / name2) as new, np.load(B / 'cells/roots40/data' / name2) as old:
                    assert set(new.files) == set(old.files) and all(same_values(new[k], old[k]) for k in old.files), 'Original feature extraction mismatch'
            metrics.extend(score_rollout(S / 'data' / name, seed, split))
            del data
        assert common.th(m.state_dict()) == state_before, 'Model parameters changed'
        assert sha(model_path) == model_hashes[str(model_path)]
        del m, checkpoint
        torch.cuda.empty_cache()
    after = verify_seal()
    assert after == before
    with (S / 'results/h0_h4_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    save(S / 'results/h0_h4_metrics.json', metrics)
    result = {'utc': utc(), 'status': 'COMPLETE_EXACT_REPLAY', 'wall_seconds': time.perf_counter() - start, 'assessment_seconds': sum(r['seconds'] for r in runs), 'gpu_peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3, 'sealed_before': before, 'sealed_after': after, 'existing_prediction_probabilities_exact': True, 'all_existing_prediction_arrays_exact': True, 'all_existing_feature_arrays_exact': True, 'probability_calls_rng_unchanged': probability_calls[0], 'weights_unchanged': model_hashes, 'original_evaluator_sha256': sha(source_path), 'new_evaluator_sha256': sha(S / 'source/evaluate_with_h2.py'), 'only_evaluator_change': 'if t in [0,2,3]: -> if t in [0,1,2,3]:', 'original_training_roots': 40, 'seeds': [0,1], 'scored_rows': 20, 'root_start_records_across_seeds': 180, 'rollout_probability_records_h0_h4': 900, 'independent_roots_total_across_splits': 30, 'no_refit': True, 'new_collection': 0, 'D_samples': 0, 'no_photometric_changes': True, 'uploaded_files': 0}
    save(S / 'results/verification.json', result)
    lines = ['# 第2步自由递推补充评估', '', '固定 roots40 joint seed0/seed1，沿用原 test/C 根与分支顺序、8次后验采样、随机种子81831、批量4、BF16和确定性设置。仅在原评估器增加第2步概率保存，不训练、不采集、不改变图像。', '', '原h0/h1/h3/h4及所有原有模式的概率、标签、顺序和元数据逐数组完全相同；319个封存文件前后SHA256校验通过。', '', '| 数据集 | 种子 | h | 四字段全对 | 五字段全对 | 九关系全对 | 四字段平均NLL | 独立根 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics:
        n = row['records']
        lines.append(f"| {row['split']} | {row['seed']} | {row['horizon']} | {row['exact4_count']}/{n} | {row['exact5_count']}/{n} | {row['relation9_count']}/{n} | {row['mean_NLL4']:.6f} | {row['independent_roots']} |")
    lines.extend(['', '四字段为红/蓝/粉对象位置和抽屉状态；五字段另加滑门。九关系为三对象Holding、InDrawer以及抽屉三种互斥状态。NLL采用自然对数，概率下限1e-9；每条记录四字段平均、每根内部平均、根间等权。', '', '同一根有三分支，h0为同一物理起点的重复记录，不是三个独立根。h0采用初始后验；h1–h4在根初始化后仅依动作连续自由递推，不接收中途真实观测。每个h为终点正确性，不是整条路径全对；h5以后未测。原始test/C已在历史诊断中使用，不构成全新盲测。', '', f"总执行耗时{result['wall_seconds']:.3f}秒；assess累计{result['assessment_seconds']:.3f}秒；GPU最大已分配{result['gpu_peak_allocated_GiB']:.3f}GiB。"])
    (S / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    files = [p for p in S.rglob('*') if p.is_file() and 'cache' not in p.relative_to(S).parts]
    save(S / 'manifest.json', {'utc': utc(), 'files': len(files), 'sha256': {str(p.relative_to(S)): sha(p) for p in sorted(files)}})
    signal.alarm(0)
    print('SUPPLEMENT_COMPLETE', json.dumps(result, ensure_ascii=False), flush=True)

MEMBERS = json.loads((B / 'configs/fixed_design.json').read_text())['members']
ALLOWED = set(MEMBERS['test10']) | set(MEMBERS['C20'])
if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if (S / 'results').exists():
            save(S / 'results/failure.json', {'utc': utc(), 'status': 'STOPPED', 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
