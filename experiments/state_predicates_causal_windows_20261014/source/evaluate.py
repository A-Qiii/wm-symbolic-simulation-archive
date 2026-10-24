from train import build
from common import *
import fcntl
E=P/'experiments/state_predicates_causal_windows_20261014';B=P/'data/predicate_causal_replay_20261014'
def joined(ds,names):
    d={k:np.concatenate([ds[n][k]for n in names])for k in ['y','v']};d['meta']=[r for n in names for r in ds[n]['meta']];return d
def main():
    assert not(E/'results/evaluation.json').exists();ds,_,roots,_=build();T=read(E/'results/training.json');fresh=joined(ds,['fresh_tune','fresh_cal']);hist=joined(ds,['old_tune','old_cal','new_dev']);out={};case_map={}
    for arm in ['ordinary_negative','hard_negative']:
        runs=[]
        for seed,r in enumerate(T[arm]['runs']):
            ps={n:np.load(E/'results'/f'{arm}_seed{seed}_{n}.npy')for n in ['old_tune','old_cal','new_dev','fresh_tune','fresh_cal']};fp=np.concatenate([ps[n]for n in ['fresh_tune','fresh_cal']]);hp=np.concatenate([ps[n]for n in ['old_tune','old_cal','new_dev']]);tau=r['calibration']['thresholds'];conditions={}
            for cond in ['hard_negative','ordinary_negative','held_positive']:
                ix=np.array([x['condition']==cond for x in fresh['meta']]);conditions[cond]=evaluate_predictions(fp[ix],{'y':fresh['y'][ix],'v':fresh['v'][ix]},tau)
            metrics={n:evaluate_predictions(p,ds[n],tau)for n,p in ps.items()};row={'seed':seed,'selected_update':r['selected_update'],'qualified':r['qualified'],'thresholds':tau,'domains':metrics,'fresh_C_all':evaluate_predictions(fp,fresh,tau),'historical_C_all':evaluate_predictions(hp,hist,tau),'fresh_conditions':conditions};runs.append(row)
            for i,m in enumerate(fresh['meta']):
                sid=m['sample_id'];q=case_map.setdefault(sid,{'metadata':m,'truth':fresh['y'][i].tolist(),'valid':fresh['v'][i].tolist(),'predictions':{},'raw_errors':[]});q['predictions'][arm+'_seed'+str(seed)]={'probabilities':fp[i].tolist(),'known':known(fp[i:i+1],tau)[0].tolist()}
                for col in np.where(fresh['v'][i]&((fp[i]>=.5)!=fresh['y'][i]))[0]:q['raw_errors'].append({'arm':arm,'seed':seed,'column':int(col),'probability':float(fp[i,col]),'truth':bool(fresh['y'][i,col])})
        out[arm]={'runs':runs,'qualified_count':sum(x['qualified']for x in runs)}
    paired=[]
    for seed,(a,b)in enumerate(zip(out['ordinary_negative']['runs'],out['hard_negative']['runs'])):
        improvements=b['fresh_conditions']['hard_negative']['held']['FP']<a['fresh_conditions']['hard_negative']['held']['FP'];fresh_fn_ok=b['fresh_C_all']['held']['FN']<=a['fresh_C_all']['held']['FN'];hist_fn_ok=b['historical_C_all']['held']['FN']<=a['historical_C_all']['held']['FN'];io_ok=all(b['historical_C_all'][g]['FP']+b['historical_C_all'][g]['FN']<=a['historical_C_all'][g]['FP']+a['historical_C_all'][g]['FN']for g in ['in_drawer','open_drawer']);paired.append({'seed':seed,'fresh_hard_FP_ordinary':a['fresh_conditions']['hard_negative']['held']['FP'],'fresh_hard_FP_hard':b['fresh_conditions']['hard_negative']['held']['FP'],'fresh_held_FN_ordinary':a['fresh_C_all']['held']['FN'],'fresh_held_FN_hard':b['fresh_C_all']['held']['FN'],'historical_held_FN_not_worse':hist_fn_ok,'historical_in_open_errors_not_worse':io_ok,'supports_preregistered_acceptance':bool(improvements and fresh_fn_ok and hist_fn_ok and io_ok)})
    out['paired']=paired;out['small_acceptance_warranted']=sum(x['supports_preregistered_acceptance']for x in paired)>=2;out['scope']='Development comparison only; no independent acceptance ran. No test-driven training or model promotion.'
    freeze={'created_utc':now(),'protocol_sha256':sha(E/'configs/protocol.json'),'small_acceptance_warranted':out['small_acceptance_warranted'],'selection':'Predeclared old+newC selection; no checkpoint replacement after this freeze','models':[{k:r[k]for k in ['arm','seed','selected_update','checkpoint_path','checkpoint_sha256','qualified']}|{'thresholds':r['calibration']['thresholds']}for arm in ['ordinary_negative','hard_negative']for r in T[arm]['runs']]};save(E/'configs/selected_models_frozen.json',freeze);(E/'configs/selected_models_frozen.json').chmod(0o444)
    failures=[]
    for q in case_map.values():
        if not q['raw_errors']:continue
        m=q['metadata'];labs=read(m['labels_private']);t=labs['frames'][-1];c=m['target_color'];q['physical_audit']={'target':c,'frames':[{'frame':f['frame'],'held':f['objects_audit'][c]['held_contact_audit'],'bilateral':f['objects_audit'][c]['bilateral_finger_force'],'external_support_contacts':f['objects_audit'][c]['external_support_contacts'],'visible_pixels':f['objects_audit'][c]['visible_pixels'],'gripper_width_m':f['gripper_width_m'],'tcp_distance_m':f['objects_audit'][c]['tcp_distance_m']}for f in labs['frames']],'motion_audit':labs['motion_audit_private']};failures.append(q)
    save(E/'results/evaluation.json',out);save(E/'results/fresh_cases.json',list(case_map.values()));save(E/'results/fresh_failures.json',failures)
    # Same final update diagnostic separates checkpoint selection from training effect.
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');final={}
    for arm in ['ordinary_negative','hard_negative']:
        final[arm]=[]
        for seed in range(3):
            m=loadmodel(E/'models'/arm/f'seed{seed}/update1500.pt');ps={n:infer(m,ds[n])for n in ['old_tune','old_cal','new_dev','fresh_tune','fresh_cal']};fp=np.concatenate([ps[n]for n in ['fresh_tune','fresh_cal']]);hp=np.concatenate([ps[n]for n in ['old_tune','old_cal','new_dev']]);conds={}
            for cond in ['hard_negative','ordinary_negative','held_positive']:
                ix=np.array([x['condition']==cond for x in fresh['meta']]);conds[cond]=evaluate_predictions(fp[ix],{'y':fresh['y'][ix],'v':fresh['v'][ix]})
            final[arm].append({'seed':seed,'fresh_conditions_raw':conds,'fresh_all_raw':evaluate_predictions(fp,fresh),'historical_all_raw':evaluate_predictions(hp,hist)});del m
    save(E/'results/fixed1500_diagnostic.json',{'purpose':'Post-freeze same-update diagnostic only; no replacement selection, calibration or training','results':final});print(json.dumps({'EVALUATED':True,'acceptance_warranted':out['small_acceptance_warranted'],'paired':paired,'fresh_failure_windows':len(failures)},ensure_ascii=False))
if __name__=='__main__':main()
