from coverage_common import *
import shutil, re, subprocess


def main():
    manifest_path = E / 'configs/delivery_manifest.json'
    assert not manifest_path.exists(), 'Already sealed; never rewrite a delivery seal.'
    verification = read(E / 'results/verification.json')
    assert verification['passed'] and verification['fits'] == 8
    attempts = read(E / 'results/training_attempts.json')
    runs = read(E / 'results/training.json')['runs']
    assert len(attempts) == len(runs) == 8
    assert all(x['status'] == 'completed' for x in attempts)
    assert all(x['updates'] == 600 for x in runs)
    expected = {(h, s, a) for h in HEADS for s in (0, 1) for a in ARMS}
    assert {(x['head'], x['seed'], x['arm']) for x in attempts} == expected
    assert {(x['head'], x['seed'], x['arm']) for x in runs} == expected
    secondary = read(E / 'results/secondary_audit.json')
    assert secondary['heads_read_on_saved_predicted_z'] == ['In']
    assert secondary['Holding_stopped_by_actual_gate']
    assert secondary['T_forward_calls'] == secondary['physics_calls'] == secondary['extra_fits'] == 0

    checked = {}
    for name in ('source_references', 'pre_encoding_frozen', 'training_frozen',
                 'final_heads_frozen', 'secondary_frozen'):
        config = read(E / f'configs/{name}.json')
        for key in ('sha256', 'validation_file_hashes'):
            for path, digest in config.get(key, {}).items():
                assert sha(path) == digest, (name, path)
                checked[path] = digest
    assert sha(E / 'source/encode.py') == read(E / 'configs/encoding_frozen.json')['source_sha256']
    for path, digest in read(E / 'results/error_review.json')['source_sha256'].items():
        assert sha(path) == digest, ('error_review_source', path)
        checked[path] = digest

    markers = {
        'prepare': 'READOUT_DESIGN_FROZEN', 'encode': 'FROZEN_ENCODING_COMPLETE',
        'train': 'EIGHT_FITS_COMPLETE', 'actual': 'ACTUAL',
        'secondary': 'SECONDARY_IN_ONLY_COMPLETE', 'verify': 'READOUT_VERIFIED',
        'report': 'READOUT_REPORT_CREATED', 'error_review': 'READONLY_ERROR_REVIEW_COMPLETE',
    }
    copied_logs = []
    for stage, marker in markers.items():
        src = Path(f'/tmp/readout_coverage_{stage}_20261018.log')
        assert marker in src.read_text(), (stage, 'missing completion marker')
        dest = E / 'logs' / src.name
        shutil.copy2(src, dest)
        copied_logs.append(str(dest))
    shutil.copy2(__file__, E / 'source/seal.py')
    commands = R / 'RUN_COMMANDS.md'
    commands.write_text(commands.read_text() + '\n收尾按补充授权运行只读错误/训练覆盖审阅，四张图人工检查后执行一次封存：\n\n```bash\npython /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/error_review.py\npython /home/robot/Documents/Codex/2026-10-13/task-4/readout_cov_src/seal.py\n```\n')

    links = []
    for markdown in R.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', markdown.read_text()):
            if target.startswith(('https:', 'http:', '#')):
                continue
            path = (markdown.parent / target.split('#')[0]).resolve()
            assert path.exists() or path == manifest_path, (markdown, target)
            links.append(str(path))
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True)
    figures = ['official_color_errors.png', 'retention_changes.png', 'Holding_red_regression.png', 'error_review_contact_sheet.png']
    assert all((R / 'figures' / x).is_file() for x in figures)
    save(E / 'results/completion_audit.json', {
        'utc': now(), 'scope': 'bounded readout data coverage control only',
        'completed_fits': 8, 'updates_each': 600, 'additional_fits': 0,
        'new_collection_or_downloads': 0, 'new_T_forward_or_fit': 0,
        'model_adoption': False, 'original_guide_completion_claim': False,
        'secondary_only_In': True, 'Holding_gate_failed': True,
        'In_C273_retention_decline_reported': True,
        'posthoc_error_review_no_model_forward': True,
        'red_FN_distinction': 'One newly wrong state and one pre-existing wrong state, from separate episodes and annotation segments; both seeds.',
        'scores_not_calibrated_probabilities': True,
        'source_and_freeze_paths_rechecked': len(checked),
        'figures_visually_inspected': figures,
        'figure_review': 'Chinese labels, axes, colors, denominator notes and images readable; legend moved to avoid covering bars.',
        'report_links_checked': len(links), 'copied_logs': copied_logs,
        'gpu_compute_processes_at_seal': gpu.stdout.strip().splitlines(),
        'no_ongoing_training_process': not gpu.stdout.strip(),
        'known_limitations': ['Only 4 independent official validation episodes.',
                              'Old D repeatedly seen; saved-z result is descriptive readout diagnosis.',
                              'Different effective positive/negative information budgets.',
                              'No further hyperparameter search or automatic model substitution.'],
    })
    files = sorted(p for root in (E, R) for p in root.rglob('*')
                   if p.is_file() and p != manifest_path)
    digests = {str(p): sha(p) for p in files}
    save(manifest_path, {'utc': now(), 'files': len(files),
                        'bytes': sum(p.stat().st_size for p in files),
                        'report_links_checked': len(links), 'sha256': digests})
    for path, digest in read(manifest_path)['sha256'].items():
        assert sha(path) == digest, path
    print('READOUT_COVERAGE_SEALED', len(files), 'files',
          sum(p.stat().st_size for p in files), 'bytes',
          'manifest_sha256', sha(manifest_path), flush=True)


if __name__ == '__main__':
    main()
