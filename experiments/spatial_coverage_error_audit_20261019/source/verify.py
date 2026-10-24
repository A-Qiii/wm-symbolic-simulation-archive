"""Read-only provenance and saved-prediction verification; --seal writes audit only."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, sys
import numpy as np

P=Path('/home/robot/research/projects/wm_symbolic')
R=P/'experiments/spatial_coverage_readout_run_20261019'
S=P/'experiments/spatial_coverage_error_audit_20261019'
COLORS=['red','blue','pink']
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def verify():
    protocol=read(S/'protocol.json'); main=read(R/'manifest.json')
    assert sha(R/'manifest.json')==protocol['source_manifest_sha256']
    assert len(main['sha256'])==main['files']==12095
    for name,h in main['sha256'].items(): assert sha(R/name)==h,name
    protected=read(R/'configs/protected_before.json')
    for name,h in protected.items(): assert sha(Path(name))==h,name
    facts=read(S/'facts.json'); cases=read(S/'selected_cases.json')
    arrays={(s,k,a):dict(np.load(R/f'data/predictions_{a}_seed{k}_{s}.npz')) for s in ['old_band','gap','externalC'] for k in [0,1] for a in ['control','coverage']}
    for f in facts:
        i=f['array_index'];j=f['object_index'];y=arrays[f['split'],0,'control']['cat'][i]
        for k in [0,1]:
            for a in ['control','coverage']:
                z=arrays[f['split'],k,a];q=z['prob'][i]
                assert str(z['root_id'][i])==f['root_id'] and str(z['episode_id'][i])==f['episode_id'] and int(z['step'][i])==f['boundary']
                if f['field'].startswith('table_'): truth=int(y[j]);pred=int(q[j*8:j*8+8].argmax())
                elif f['field'] in ['inside_drawer','outside_drawer']: truth=bool(y[j]==3);pred=bool(q[3+j*8]>=.5)
                elif 'identity' in f['field']: truth=[c for n,c in enumerate(COLORS) if y[n]==6];pred=[c for n,c in enumerate(COLORS) if q[6+n*8]>=.5]
                else: truth=int(y[3]);pred=int(q[24:].argmax())
                assert f['truth']==truth and f[f'{a}_seed{k}']==pred
        old=[f[f'control_seed{k}']==f['truth'] for k in [0,1]];new=[f[f'coverage_seed{k}']==f['truth'] for k in [0,1]]
        category='common_fixed' if not any(old) and all(new) else 'common_unfixed' if not any(old) and not any(new) else 'common_regression' if all(old) and not any(new) else 'seed_discordant_regression' if any(o and not n for o,n in zip(old,new)) else 'other'
        assert category==f['category']
    counts=list(csv.DictReader((S/'case_counts.csv').open()))
    for row in counts:
        q=[f for f in facts if f['split']==row['split'] and f['field']==row['field'] and f['category']==row['category'] and (row['cohort']!='h0' or f['h0'])]
        assert len(q)==int(row['facts']) and len({f['root_id'] for f in q})==int(row['independent_roots'])
    assert len(cases)==len({f['root_id'] for f in cases})==10
    for case in cases:
        assert any(all(case[k]==f[k] for k in ['split','array_index','field','color','truth','category']) for f in facts)
        assert (S/'figures'/case['figure']).exists()
    cover=read(S/'C_coverage_records.json'); summary=read(S/'C_coverage_summary.json')
    for row in summary:
        q=[f for f in cover if f['scope']==row['scope'] and f['truth']==row['class'] and (row['subset']=='all' or f['both_wrong'])]
        assert len(q)==row['objects'] and len({f['root_id'] for f in q})==row['roots']
        for k in ['within_pooled_x','within_same_color_x','within_same_color_xy_hull','within_same_color_static_hull','static_visible']:
            assert sum(f[k] for f in q)==row[k]
        assert sum(f['availability']>0 for f in q)==row['public_roi_available']
    assert len(facts)==482 and len(counts)==168
    assert not any(f['h0'] and f['category']=='common_fixed' for f in facts)
    for name in ['compute_complete.json','report_complete.json']:
        receipt=read(S/name)
        for k in ['new_forwards','new_fits','new_simulations']: assert receipt[k]==0
    assert 'torch' not in sys.modules and 'pybullet' not in sys.modules
    assert (S/'REPORT.html').read_text().count('data:image/png;base64,')==12
    return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','main_sealed_files_unchanged':len(main['sha256']),'main_manifest_sha256':sha(R/'manifest.json'),'older_sealed_files_unchanged':len(protected),'saved_prediction_facts_verified':len(facts),'count_rows_recomputed':len(counts),'selected_distinct_roots':len(cases),'coverage_summary_rows_recomputed':len(summary),'figures':12,'new_model_forwards':0,'new_fits':0,'new_simulations':0,'uploads':0}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    result=verify()
    if args.seal:
        assert not (S/'manifest.json').exists(),'Do not reseal an existing audit'
        save(S/'final_verification.json',result)
        entries={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts and p.name!='manifest.json'}
        save(S/'manifest.json',{'files':len(entries),'sha256':entries})
    else:
        m=read(S/'manifest.json')
        for name,h in m['sha256'].items(): assert sha(S/name)==h,name
        result['audit_sealed_files_unchanged']=len(m['sha256'])
    print(json.dumps(result,ensure_ascii=False))
