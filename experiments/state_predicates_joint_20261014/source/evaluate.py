from common import *
import fcntl,csv
def main():
    assert read(E/'results/acceptance_qa.json')['passed'];freeze=read(E/'results/candidate_freeze.json');assert not(E/'results/evaluation.json').exists()
    for m in freeze['models']:assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');data={}
    om=[r for r in read(O/'data/metadata.json')if r['split']=='diagnostic_D']
    with np.load(O/'data/inputs_targets.npz')as z:
        sel=z['splits']=='diagnostic_D';yy,vv=targets(torch.tensor(z['cat'][sel]),torch.tensor(z['valid'][sel]));data['old_D']={k:z[k][sel].copy()for k in ['objects','context','robot']};data['old_D'].update(y=yy.numpy(),v=vv.numpy(),meta=om,roots=np.array([r['root_id']for r in om]))
    for cohort,path,meta in [('seen_test',A/'data/supplement_test.npz',A/'data/metadata_test.json'),('acceptance',E/'data/supplement_acceptance.npz',E/'data/metadata_acceptance.json')]:
        mm=read(meta)
        with np.load(path)as z:data[cohort]={k:z[k].copy()for k in ['objects','context','robot']};data[cohort].update(y=z['y'].copy(),v=z['eval_valid'].copy(),meta=mm,roots=np.array([r['root_id']for r in mm]))
    candidates=[]
    for seed in range(3):
        candidates.append({'name':f'original_seed{seed}','path':O/'models'/f'direct_predicates_seed{seed}.pt','thresholds':[.5]*3,'primary':False})
        candidates.append({'name':f'prior_seed{seed}','path':A/'models'/f'augmented_seed{seed}_selected.pt','thresholds':read(A/'results/model_threshold_freeze.json')['models'][f'augmented_seed{seed}']['thresholds'],'primary':False})
        for family in ['in_supervision','balanced_loss']:
            r=read(E/'results'/f'{family}.json')['seeds'][seed];candidates.append({'name':f'{family}_seed{seed}','path':Path(r['checkpoint_path']),'thresholds':r['calibration']['thresholds'],'primary':family=='balanced_loss'and r['qualified']})
    results={};rows=[]
    for c in candidates:
        m=loadmodel(c['path']);res={'checkpoint':str(c['path']),'checkpoint_sha256':sha(c['path']),'thresholds':c['thresholds'],'primary_frozen_candidate':c['primary']}
        for cohort,d in data.items():
            if cohort=='acceptance'and not(c['name'].startswith('balanced_loss')or c['name'].startswith('original')):continue
            pp=infer(m,d);np.save(E/'results'/f"{c['name']}_{cohort}.npy",pp);met=evaluate_predictions(pp,d,c['thresholds']);res[cohort]=met
            groups={}
            if cohort!='old_D':
                for cond in sorted({r['condition']for r in d['meta']}):
                    inds=np.array([r['condition']==cond for r in d['meta']]);groups[cond]={'windows':int(inds.sum()),'roots':len(set(d['roots'][inds])),'metrics':summarize(pp[inds],d['y'][inds],d['v'][inds],c['thresholds'])}
                res[cohort+'_conditions']=groups
            else:
                near=[i for i,r in enumerate(d['meta'])if r['index']in[1283,1284]];mask=np.zeros_like(d['v']);mask[near,0]=True;res['old_D_original_near']=metric(pp,d['y'],mask,known(pp,c['thresholds']))
            kk=known(pp,c['thresholds'])
            for i,j in np.argwhere(d['v']):
                r=d['meta'][i];rows.append({'model':c['name'],'cohort':cohort,'sample_id':r.get('sample_id',str(r.get('index'))),'root_id':r['root_id'],'predicate_col':int(j),'truth':bool(d['y'][i,j]),'probability':float(pp[i,j]),'raw_prediction':bool(pp[i,j]>=.5),'known':bool(kk[i,j])})
            if cohort=='acceptance':
                gate=raw_gate({'acceptance':met});knownpass=all(q['coverage']>=.95 and(q['positive']==0 or q['known_true_positive']/q['positive']>=.95)and(q['negative']==0 or q['known_false_positive']/q['negative']<=.01)for q in met.values());res['acceptance_gate']={'raw':gate,'known_constraints_pass':knownpass,'full_positive_coverage':read(E/'results/acceptance_qa.json')['full_positive_coverage'],'passed':gate['passed']and knownpass and read(E/'results/acceptance_qa.json')['full_positive_coverage']}
        results[c['name']]=res;del m
    with(E/'results/predictions.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out={'created_utc':now(),'frozen_qualified_seeds':freeze['qualified_seeds'],'models':results,'cohorts':{n:{'windows':len(d['y']),'roots':len(set(d['roots']))}for n,d in data.items()},'no_test_based_selection_or_repair':True,'acceptance_run_once_after_candidate_freeze':True,'primary_passed_seeds':[int(n.rsplit('seed',1)[1])for n,r in results.items()if r['primary_frozen_candidate']and r['acceptance_gate']['passed']]}
    save(E/'results/evaluation.json',out)
    print(json.dumps({'cohorts':out['cohorts'],'primary_passed_seeds':out['primary_passed_seeds'],'acceptance':{n:{g:{k:q[k]for k in ['positive','negative','FP','FN','unknown']}for g,q in r['acceptance'].items()}for n,r in results.items()if'acceptance'in r}},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
