from common import *
import re
import shutil
import subprocess
from collections import Counter


def main():
    manifest = E / 'configs/delivery_manifest.json'
    assert not manifest.exists(), 'Already sealed; use read-only manifest verification.'
    phases = {}
    for name in ['pre_annotation_frozen', 'annotations_frozen', 'source_references',
                 'feature_code_frozen', 'training_frozen', 'final_heads_frozen']:
        f = E / f'configs/{name}.json'
        entries = read(f)['sha256']
        for p, h in entries.items():
            assert sha(p) == h, p
        phases[name] = {'files_checked': len(entries), 'manifest_sha256': sha(f)}
    prior = {}
    for name in ['visual_node_edge_readout_20261018',
                 'visual_edge_zero_context_control_20261018',
                 'endpoint_rgb_feasibility_20261018']:
        f = P / f'experiments/{name}/configs/delivery_manifest.json'
        entries = read(f)['sha256']
        for p, h in entries.items():
            assert sha(p) == h, p
        prior[name] = {'files_checked': len(entries), 'manifest_sha256': sha(f)}

    gate = read(E / 'results/annotation_quality_gate.json')
    verification = read(E / 'results/verification.json')
    attempts = read(E / 'results/training_attempts.json')
    training = read(E / 'results/training.json')
    assert gate['passed'] and verification['passed']
    assert len(attempts) == 2 and all(x['status'] == 'completed' for x in attempts)
    assert [x['seed'] for x in attempts] == [0, 1]
    assert training['fits'] == 2 and training['DINO_updates'] == 0
    assert all(x['updates'] == 600 and x['parameters'] == 25290 for x in training['runs'])
    assert not training['validation_access'] and not training['model_selection']
    assert len(list((E / 'models').glob('*update*.pt'))) == 2
    supcon = P / 'experiments/readout_supcon_ablation_20261018'
    assert not list(supcon.rglob('training_attempts.json'))
    assert not list(supcon.rglob('*update0600.pt'))

    rows = read(E / 'data/public/images.json')
    annotations = read(E / 'data/annotations/final.json')['annotations']
    assert len(rows) == len(annotations) == 40
    assert len({x['state_id'] for x in rows}) == 20
    by_id = {x['image_id']: x for x in rows}
    statuses = Counter(c['status'] for a in annotations for c in a['classes'])
    box_count = sum(len(c['boxes']) for a in annotations for c in a['classes'])
    wrist_front_unknown = sum(
        by_id[a['image_id']]['camera'] == 1 and c['category'] == 'drawer_front'
        and c['status'] == 'uncertain' for a in annotations for c in a['classes'])
    assert box_count == 91 and statuses['not_visible'] == 0 and wrist_front_unknown == 20
    assert all(not c['boxes'] for a in annotations for c in a['classes']
               if c['status'] == 'uncertain')
    ev = read(E / 'results/evaluation.json')
    assert all(x['TP'] is None and x['FP'] is None and x['FN'] is None
               for x in ev['records'] if x['status'] == 'uncertain')
    scale = read(E / 'results/scale_and_coverage_audit.json')
    assert not scale['local_DINO_resolution_code']['448_inference_or_training_run']
    code = scale['local_DINO_resolution_code']
    assert sha(code['path']) == code['sha256']

    log_names = ['prepare', 'sheets', 'round1', 'overlays', 'review', 'quality',
                 'features', 'freeze', 'train', 'evaluate', 'diagnose', 'verify',
                 'report', 'scale', 'scale_initial_syntax']
    (E / 'logs').mkdir(exist_ok=True)
    for name in log_names:
        f = Path(f'/tmp/endpoint_box_{name}_20261018.log')
        assert f.is_file(), f
        shutil.copy2(f, E / 'logs' / f.name)
    shutil.copy2(__file__, E / 'source/seal.py')
    commands = R / 'RUN_COMMANDS.md'
    text = commands.read_text()
    assert 'scale_audit.py' not in text
    commands.write_text(text + '\nreport.py后另行执行scale_audit.py，只统计已冻结框尺寸、相邻中心和本地DINO位置插值实现，不作新推理或训练。第一次调用因报告字符串语法错误在执行前停止，修正后一次完成；原错误日志一并保留，不属于模型重试。随后seal.py逐文件复核此前冻结与旧实验、复制日志并生成交付哈希。\n\n封存命令（已执行，不应在封存目录再次运行）：\n\n```bash\nsource /home/robot/research/projects/wm_symbolic/world_model_env.sh\nPYTHONDONTWRITEBYTECODE=1 python /home/robot/Documents/Codex/2026-10-13/task-4/endpoint_box_src/seal.py\n```\n')

    links_checked = 0
    for f in [R / 'REPORT.md', R / 'SCALE_AND_NEXT_STEP.md', commands]:
        for target in re.findall(r'\]\(([^)]+)\)', f.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            target_path = (f.parent / target.split('#')[0]).resolve()
            assert target_path == manifest or target_path.is_file(), (f, target)
            links_checked += 1
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader'], text=True, capture_output=True)
    assert gpu.returncode == 0, gpu.stderr
    audit = {
        'utc': now(), 'status': 'completed_bounded_annotation_and_localization_diagnostic',
        'overall_world_model_guide_completed': False, 'new_RGB_images': 40,
        'selected_existing_states': 20, 'known_visible_boxes': box_count,
        'model_fits': 2, 'updates_each': 600, 'DINO_updates': 0,
        'T_updates': 0, 'SupCon_updates': 0, 'relation_training': False,
        'new_downloads_or_data_collection': False, 'model_adopted': False,
        'not_visible_class_images': statuses['not_visible'],
        'wrist_drawer_front_uncertain_images': wrist_front_unknown,
        'uncertain_classes_neither_negative_nor_scored': True,
        'annotation_quality_is_same_assistant_repeat_not_independent_accuracy': True,
        'source_split_role': 'AB heldout development; all sources historically seen; C/D diagnostic only',
        'AB_development_TP_GT': {'drawer_front': [[4, 8], [3, 8]],
                                 'gripper_finger': [[5, 29], [2, 29]]},
        'single_component_diagnostic': 'Saved logits only: remove local-peak filter; training recall improves, AB recall unchanged; not adopted',
        'scale_evidence': '7/14 training static finger boxes shorter than one patch; larger wrist boxes also fail; resolution is not isolated causally',
        'next_step_recommendation_only': 'Independent annotation review, then potentially frozen 224/448 comparison; no new inference or training executed',
        'figures_reviewed': ['All 40 raw RGB and initial annotation overlays during annotation',
                             'Corrected finger crops and RGB-only repeat sheet',
                             'localization_recall.png', 'prediction_examples.png',
                             'final_annotations_page_2.png', 'final_annotations_page_4.png'],
        'markdown_links_checked': links_checked, 'phase_hash_checks': phases,
        'prior_deliveries_unchanged': prior,
        'gpu_compute_processes_at_seal': gpu.stdout.strip().splitlines(),
        'preserved_non_model_error': 'Report-only scale script initial syntax error; no code executed, corrected before single successful audit',
        'preserved_preflight_warning': 'Tensor-to-float warning during masked-class gradient check; check passed',
    }
    save(E / 'results/completion_audit.json', audit)
    save(E / 'logs/seal_checks.json', {'utc': now(), 'passed': True,
         'phase_files_checked': sum(x['files_checked'] for x in phases.values()),
         'prior_delivery_files_checked': sum(x['files_checked'] for x in prior.values()),
         'links_checked': links_checked, 'training_not_repeated': True})
    files = sorted(p for base in [E, R] for p in base.rglob('*') if p.is_file() and p != manifest)
    hashes = {str(p): sha(p) for p in files}
    sizes = {str(p): p.stat().st_size for p in files}
    save(manifest, {'utc': now(), 'status': 'sealed', 'files': len(files),
                    'bytes': sum(sizes.values()), 'sha256': hashes, 'sizes': sizes,
                    'manifest_excludes_itself': True,
                    'scope': '40RGB box annotation and two-seed frozen-DINO localization diagnostic; failed readiness, no adoption'})
    for p, h in read(manifest)['sha256'].items():
        assert sha(p) == h, p
    print('ENDPOINT_BOX_DELIVERY_SEALED', len(files), sum(sizes.values()), sha(manifest), flush=True)


if __name__ == '__main__':
    main()
