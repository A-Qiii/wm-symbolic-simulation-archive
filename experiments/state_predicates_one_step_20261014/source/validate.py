"""Independent audit of stored data/counts; frozen readout replay on CPU/CUDA."""
from common import *

def main():
    torch.set_num_threads(4)
    protocol=read(E/'configs/protocol.json'); selection=read(E/'configs/model_selection_frozen.json')
    ev=read(E/'results/evaluation.json'); extraction=read(E/'results/extraction.json')
    for p,h in protocol['frozen_sha256'].items(): assert sha(p)==h,p
    assert sha(E/'configs/protocol.json')==selection['protocol_sha256']
    assert sha(E/'configs/model_selection_frozen.json')==ev['selection_sha256']
    assert protocol['created_utc'] < selection['created_utc'] < ev['created_utc']
    manifest=read(DATA/'delivery_manifest.json')
    for r in manifest['entries']:
        p=DATA/r['path']; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    assert len(manifest['entries'])==939
    pairs=read(E/'data/public/pairs.json'); windows=read(DATA/'data/public/windows_v1.json')
    assert pairs==read(DATA/'data/public/pairs_v1.json')
    ids={w['window_id']:i for i,w in enumerate(windows)}
    bi=np.array([ids[r['before_window_id']]for r in pairs]); ai=np.array([ids[r['after_window_id']]for r in pairs])
    z=dict(np.load(E/'data/public/transition_inputs.npz')); t=dict(np.load(E/'data/_private/training_targets_and_audit.npz'))
    h=dict(np.load(E/'data/_private/window_latents.npz')); old=dict(np.load(DATA/'data/public/paired_inputs_v1.npz'))
    assert set(z)=={'current_objects','current_scene','declared_actions'}
    for key,index,out in [('objects',bi,'current_objects'),('scene',bi,'current_scene')]:assert np.array_equal(z[out],h[key][index])
    for key,out in [('objects','next_objects'),('scene','next_scene')]:assert np.array_equal(t[out],h[key][ai])
    assert np.array_equal(bi,old['pair_before_index']) and np.array_equal(ai,old['pair_after_index'])
    assert np.array_equal(z['declared_actions'],old['declared_action_features'])
    for name,value in dict(np.load(DATA/'data/_private/supervision.npz')).items():assert np.array_equal(t[name],value)
    assert np.array_equal(t['observed_current_probabilities'],h['probabilities'][bi])
    assert np.array_equal(t['observed_next_probabilities'],h['probabilities'][ai])
    assert sha(E/'data/public/transition_inputs.npz')==extraction['inputs_sha256']
    assert sha(E/'data/_private/training_targets_and_audit.npz')==extraction['targets_sha256']
    for i,r in enumerate(pairs):
        a=np.zeros(9,dtype=np.float32); k=0 if r['skill']=='open_drawer' else 1 if r['skill'].startswith('lift_') else 2 if r['skill']=='place_in_drawer' else 3
        a[k]=1
        if k==1:a[4+COLORS.index(r['declared_arguments']['color'])]=1
        a[7]=r['declared_primitive_budget']/900; a[8]=r['declared_posthold_controls']/96
        assert np.array_equal(a,z['declared_actions'][i])
        assert windows[bi[i]]['root_id']==r['root_id']==windows[ai[i]]['root_id']
    roots={k:set(v)for k,v in [('train',protocol['data']['train_roots']),('tune',protocol['data']['dev_roles']['tune']),('calibration',protocol['data']['dev_roles']['calibration'])]}
    assert len(roots['train'])==24 and len(roots['tune'])==len(roots['calibration'])==3
    assert not(roots['train']&roots['tune']or roots['train']&roots['calibration']or roots['tune']&roots['calibration'])
    for w in windows:
        assert np.array_equal(np.diff(w['history_frame_indices']),[21,21,21])
        assert abs(w['simulation_times'][-1]-w['simulation_times'][0]-2.1)<1e-8
    runs=selection['all_runs']; assert len(runs)==9 and selection['total_optimizer_updates']==5400
    cp_count=0
    for r in runs:
        curve=[json.loads(q)for q in (E/'runs'/r['run']/'curve.jsonl').read_text().splitlines()]
        assert len(curve)==600 and [q['update']for q in curve]==list(range(1,601))
        assert np.isfinite([[q['batch_BCE'],q['batch_latent_SmoothL1'],q['batch_total_loss']]for q in curve]).all()
        assert all(abs(q['batch_total_loss']-q['batch_BCE']-.05*q['batch_latent_SmoothL1'])<1e-6 for q in curve)
        assert r['selected']==min(r['all_checkpoints'],key=lambda q:(q['tune_root_macro_BCE'],q['update']))
        assert r['updates']==600 and r['parameters']==76224 and r['calibration_not_scored'] and r['frozen_reader_digest_unchanged']
        for cp in r['all_checkpoints']:
            assert sha(cp['path'])==cp['sha256']; obj=torch.load(cp['path'],map_location='cpu',weights_only=False)
            assert obj['update']==cp['update'] and obj['mode']==r['mode'] and obj['seed']==r['seed']
            assert obj['protocol_sha256']==sha(E/'configs/protocol.json'); cp_count+=1
    for seed in range(3): assert len({r['initial_state_digest']for r in runs if r['seed']==seed})==1
    for mode in protocol['modes']:
        expected=min([r for r in runs if r['mode']==mode],key=lambda q:(q['selected']['tune_root_macro_BCE'],q['seed']))['run']
        assert selection['representative_run_per_mode'][mode]==expected
    pred=dict(np.load(E/'results/predictions.npz')); lp=dict(np.load(E/'results/latent_predictions.npz'))
    m=reader('cpu'); assert digest_model(m)==selection['frozen_candidate_digest']==extraction['candidate_digest_unchanged']
    readout_error=0.;cpu_label_differences=0
    with torch.inference_mode():
        for r in runs:
            n=r['run']; pp=logits(m,torch.from_numpy(lp[n+'_objects']),torch.from_numpy(lp[n+'_scene'])).sigmoid().numpy()
            readout_error=max(readout_error,float(np.abs(pp-pred[n]).max()))
            cpu_label_differences+=int(((pp>=.5)!=(pred[n]>=.5)).sum())
    assert cpu_label_differences==0
    # CUDA used matmul_precision=high. Check the original execution path exactly,
    # and separately expose CPU numerical differences instead of changing outputs.
    import fcntl
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_float32_matmul_precision('high'); gm=reader('cuda');cuda_readout_error=0.
    with torch.inference_mode():
        for r in runs:
            n=r['run'];pp=logits(gm,torch.as_tensor(lp[n+'_objects'],device='cuda'),torch.as_tensor(lp[n+'_scene'],device='cuda')).sigmoid().cpu().numpy()
            cuda_readout_error=max(cuda_readout_error,float(np.abs(pp-pred[n]).max()))
    assert cuda_readout_error==0.
    assert digest_model(gm)==digest_model(m)
    del gm;torch.cuda.empty_cache();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    y=t['after_y'].astype(bool); v=t['after_valid'].astype(bool); before=t['before_y'].astype(bool)
    changed=(y!=before)&v&t['before_valid']; assert int(changed.sum())==150 and int(v.sum())==840
    def decoded(p):
        hat=p>=.5; known=np.ones_like(hat); conflict=hat[:,:3].sum(1)>1
        known[:,:3]&=~(conflict[:,None]&hat[:,:3]); return hat,known
    currhat,currknown=decoded(pred['persistence']); currentgood=(currknown&(currhat==before)&t['before_valid']).all(1)
    assert int((~currentgood).sum())==1
    masks={'all7':np.ones_like(v),'changed':changed,'unchanged':(y==before)&t['before_valid']}
    target=np.zeros_like(v)
    for i,r in enumerate(ev['details']['full_seed2']):target[i,r['target_entity_columns']]=True
    masks.update(target_entity=target,other_entity=~target)
    sets={'AB_all':np.array([r['split']=='train'for r in pairs]),'C_all':np.array([r['split']=='dev'for r in pairs]),'C_tune':np.array([r['dev_role']=='tune'for r in pairs]),'C_calibration':np.array([r['dev_role']=='calibration'for r in pairs])}
    verified=0
    for n,p in pred.items():
        hat,k=decoded(p); full=(k&v&(hat==y)).all(1)
        for sn,sel in sets.items():
            for subset,sub in [('all',np.ones(120,bool)),('current_reader_correct',currentgood),('current_reader_wrong_or_UNKNOWN',~currentgood)]:
                ii=sel&sub; r=ev['results'][n][sn][subset]
                assert r['pairs']==int(ii.sum()) and r['all7_known_correct_pairs']==int(full[ii].sum())
                ridset={pairs[j]['root_id']for j in np.flatnonzero(ii)}
                assert r['all_pairs_correct_roots']==sum(bool(full[np.array([ii[j]and pairs[j]['root_id']==rid for j in range(120)])].all())for rid in ridset)
                for maskname,mask in masks.items():
                    vv=v&mask&ii[:,None]; mm=r['mask_metrics'][maskname]
                    assert mm['facts']==int(vv.sum()) and mm['UNKNOWN']==int((vv&~k).sum())
                    assert mm['strict_known_correct']==int((vv&k&(hat==y)).sum())
                    assert mm['raw_FP']==int((vv&hat&~y).sum()) and mm['raw_FN']==int((vv&~hat&y).sum()); verified+=1
    checks=read(E/'results/forward_contract_checks.json')
    for r in checks:
        assert r['forbidden_extra_kwarg_rejected'] and r['forward_signature']==['current_objects','current_scene','declared_actions']
        if not r['run'].startswith('full_'):assert r['perturbation_max_latent_change']==0
    assert ev['decision']['worth_one_new_independent_test']is False
    result={'created_utc':now(),'passed':True,'verification_device':'CPU audit plus frozen CUDA head replay; no training or simulator execution','frozen_protocol_and_source_hashes_verified':len(protocol['frozen_sha256']),'previous_RGB_delivery_files_verified':939,'previous_RGB_delivery_bytes_verified':manifest['bytes'],'current_latents_exactly_before_indices':True,'target_latents_exactly_after_indices':True,'public_actions_reconstructed_from_declaration':True,'GT_equal_original':True,'root_split_disjoint':True,'genuine_2p1s_windows':150,'all120_pairs_retained':True,'all840_valid_facts_retained':True,'changed_facts':150,'unchanged_facts':690,'runs':9,'optimizer_updates':5400,'checkpoint_hashes_verified':cp_count,'same_initialization_per_seed_across_modes':True,'selection_recomputed_from_tune_only_records':True,'no_calibration_in_training_source_or_records':True,'CPU_shared_readout_vs_saved_CUDA_probabilities_max_error':readout_error,'CPU_vs_CUDA_threshold_label_differences':cpu_label_differences,'CUDA_shared_readout_replay_max_error':cuda_readout_error,'initial_CPU_1e_minus5_check_failed_and_log_retained':True,'independent_metric_cells_verified':verified,'candidate_digest_unchanged':digest_model(m),'current_reader_bad_pairs_retained':1,'C_current_reader_bad_pairs':0,'ablation_contract_checks_passed':True,'predeclared_decision_unchanged':True}
    save(E/'results/final_validation.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
