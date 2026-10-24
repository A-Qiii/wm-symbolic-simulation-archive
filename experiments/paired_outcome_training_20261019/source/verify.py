from study import *

def main():
    guard('verify')
    assert not (U/'SEALED.json').exists()
    frozen=read(U/'configs/training_frozen.json')
    for p,h in frozen['sha256'].items():
        assert sha(p)==h, p
    historical=[]
    for folder in [R,I]:
        seal=read(folder/'SEALED.json'); manifest=read(folder/'MANIFEST.json')
        assert sha(folder/'MANIFEST.json')==seal['manifest_sha256']
        for row in manifest['files']:
            p=folder/row['path']; assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
        historical.append({'path':str(folder),'files':len(manifest['files']),'manifest_sha256':seal['manifest_sha256'],'all_unchanged':True})
    node=next(n for n in ast.parse((R/'source/train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in node.body if isinstance(n,ast.For) and any(isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='arm' for x in ast.walk(n)))
    assert (U/'source/exact_original_training_loop.py').read_text()==ast.unparse(loop)+'\n'
    reuse=read(U/'results/control_reuse_audit.json')
    assert reuse['reuse_allowed'] and len(reuse['checks'])==4
    assert all(c['stored_dev_objective']==c['recomputed_dev_objective'] for c in reuse['checks'])
    models=read(U/'configs/models_frozen.json'); complete=read(U/'results/training_complete.json')
    assert complete['fits']==2 and complete['updates']==2000 and complete['new_collection']==0 and complete['C_D_samples']==0 and complete['control_refits']==0
    for p,h in models['sha256'].items(): assert sha(p)==h,p
    for seed in [0,1]:
        assert sha(U/f'configs/schedule_seed{seed}.npz')==sha(R/f'configs/schedule_seed{seed}.npz')
        with np.load(U/f'configs/schedule_seed{seed}.npz') as z:
            assert z['old'].shape==(1000,2,4) and z['new'].shape==(1000,2,4)
            assert (np.bincount(z['new'].reshape(-1),minlength=24)>0).all()
        new=read(U/f'results/train_paired_seed{seed}.json');old=read(R/f'results/train_recovery_seed{seed}.json')
        assert new['initial_hash']==old['initial_hash'] and new['updates']==old['updates']==1000
        assert new['old_samples']==new['new_samples']==old['old_samples']==old['new_samples']==8000
        assert [c['update'] for c in new['candidates']]==[500,1000]
        assert new['selected']==min(new['candidates'],key=lambda c:(c['development_objective'],c['update']))
        assert all(np.isfinite(r['objective']) for r in new['curve'])
    sub=read(U/'configs/branch_substitutions.json')
    assert len(sub)==24
    assert sum(r['control_branch']==0 and r['paired_branch']==1 and r['changed_action_indices']==[1,3] for r in sub)==12
    assert sum(r['control_branch']==2 and r['paired_branch']==2 and r['changed_action_indices']==[] for r in sub)==12
    ef=read(U/'configs/evaluation_frozen.json')
    for key,path in [('protocol_sha256',U/'configs/protocol.json'),('legacy_evaluator_sha256',R/'source/evaluate.py'),('local_evaluator_sha256',U/'source/evaluate.py'),('analysis_sha256',U/'source/analyze.py')]:assert ef[key]==sha(path)
    assert frozen['utc']<models['utc']<ef['utc']
    infer=read(U/'results/inference_complete.json')
    assert infer['model_weights_unchanged'] and not infer['new_blind_evaluation'] and not infer['new_training_during_evaluation']
    for receipt in infer['reports']:
        assert receipt['paired_prefault_probabilities_exact'] and receipt['suffix_future_observations_used']==0 and receipt['samples']==8
        assert receipt['prediction_rows']==66*receipt['roots']
    refs=read(U/'results/prediction_references.json');assert len(refs)==24
    arrays={}
    for ref in refs:
        assert sha(ref['path'])==ref['sha256']
        with np.load(ref['path']) as z:d={k:z[k] for k in z.files}
        assert len(d['prob'])==(792 if ref['split']=='train' else 396)
        assert np.isfinite(d['prob']).all() and ((d['prob']>=0)&(d['prob']<=1)).all()
        assert d['cat_valid'][:,:4].all()
        for rid in set(d['root_id']):
            ix=lambda b:np.where((d['root_id']==rid)&(d['branch']==b)&(d['mode']=='one_step')&(d['step']==2))[0][0]
            assert np.array_equal(d['prob'][ix(1)],d['prob'][ix(2)])
        arrays[ref['variant'],ref['seed'],ref['split']]=d
    for seed in [0,1]:
        for split in ['train','development','evaluation']:
            reference=arrays['control',seed,split]
            for variant in ['paired','control_final1000','paired_final1000']:
                other=arrays[variant,seed,split]
                for k in ['root_id','episode_id','branch','mode','step','cat','cat_valid','status','skill','controls']:
                    assert np.array_equal(reference[k],other[k]),(variant,seed,split,k)
    metrics=read(U/'results/analysis_complete.json')['metrics']
    def count(v,s,sp,m):return next(r['sum'] for r in metrics if (r['variant'],r['seed'],r['split'],r['metric'])==(v,s,sp,m))
    gates=read(U/'results/gates.json');assert not gates['full_guide_acceptance']
    for row in gates['gates']:
        s=row['seed'];assert row['training_necessary_gate']==(count('paired',s,'train','h0_held')==12 and count('paired',s,'train','h2_pair_joint')==12)
        assert row['development_support_gate']==all(row['development_components'].values())
        assert row['historical_evaluation_not_used_for_gate']
    assert all(np.isfinite(r['mean']) and np.isfinite(r['ci_low']) and np.isfinite(r['ci_high']) for r in metrics)
    assert len(list((U/'figures').glob('*.png')))==4
    report=read(U/'results/report_complete.json');assert report['figures']==4
    assert '历史评估' in (U/'REPORT.md').read_text() and '纯粹的配对标签效应' in (U/'REPORT.md').read_text()
    verification={'utc':now(),'passed':True,'training_input_hashes_unchanged':len(frozen['sha256']),'historical_seals':historical,'exact_original_loop_reused':True,'old_control_reuse_verified':True,'two_new_fits_only':True,'same_sampling_schedule_and_initialization':True,'new_models_unchanged_since_selection':True,'all24_prediction_references_verified':True,'same_scoring_labels_and_order':True,'paired_prefault_predictions_exact':True,'gates_not_relaxed':True,'historical_evaluation_exploratory_only':True,'C_D_samples':0,'new_collection':0,'full_guide_acceptance':False}
    save(U/'results/verification.json',verification)
    files=[]
    for p in sorted(U.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and 'cache' not in p.relative_to(U).parts and p.name not in ['MANIFEST.json','SEALED.json']:
            files.append({'path':str(p.relative_to(U)),'bytes':p.stat().st_size,'sha256':sha(p)})
    save(U/'MANIFEST.json',{'utc':now(),'files':files})
    save(U/'SEALED.json',{'utc':now(),'manifest_sha256':sha(U/'MANIFEST.json'),'files':len(files),'bytes':sum(f['bytes']for f in files),'verification_passed':True,'no_further_training_or_selection':True})
    print(json.dumps(read(U/'SEALED.json')),flush=True)

if __name__=='__main__':main()
