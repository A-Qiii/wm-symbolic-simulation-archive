from common import *
from pair_model import PairReader
import fcntl,time,random
import torch.nn.functional as F
E=P/'experiments/state_predicates_relations_20261014';J=P/'experiments/state_predicates_joint_20261014'
def pred(m,d):
    m.eval()
    with torch.no_grad():return m.probabilities(**{k:torch.as_tensor(d[k],device='cuda')for k in ['objects','context','robot','neighbors']}).cpu().numpy()
def main():
    assert not(E/'results/pair_training.json').exists();ds=dataset()
    with np.load(E/'data/neighbors_train_dev.npz')as z:
        for n in ds:ds[n]['neighbors']=z[n].copy()
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    dat={k:torch.tensor(np.concatenate([ds[n][k]for n in ['old_train','new_train']]),device='cuda')for k in ['objects','context','robot','neighbors']};yy=np.concatenate([ds[n]['y']for n in ['old_train','new_train']]);vv=np.concatenate([ds[n]['v']for n in ['old_train','new_train']]);y=torch.tensor(yy,device='cuda');v=torch.tensor(vv,device='cuda');weights=read(J/'configs/balanced_loss_loss.json')['weights'];tune={n:ds[n]for n in ['old_tune','new_dev']};cal={n:ds[n]for n in ['old_cal','new_dev']}
    torch.manual_seed(0);base=PredicateReader().cuda().eval();torch.manual_seed(0);probe=PairReader(True).cuda().eval();batch={k:z[:2]for k,z in dat.items()}
    with torch.no_grad():
        p0=base.probabilities(**{k:batch[k]for k in ['objects','context','robot']});p1=probe.probabilities(**batch);initerr=float((p0-p1).abs().max());probe.fusion.weight.normal_(0,.01);q=probe.probabilities(**batch);nb={**batch,'neighbors':batch['neighbors'][:,:,:,torch.arange(11,-1,-1,device='cuda')]};perr=float((q-probe.probabilities(**nb)).abs().max());ob={**batch,'objects':batch['objects'][:,:,[2,0,1]]};oerr=float((probe.probabilities(**ob)-q[:,[2,0,1,5,3,4,6]]).abs().max())
    assert initerr<1e-6 and perr<1e-6 and oerr<1e-6
    params=sum(p.numel()for p in probe.parameters());save(E/'results/model_checks.json',{'zero_residual_equals_original':initerr,'neighbor_permutation_error':perr,'target_permutation_error':oerr,'base_parameters':sum(p.numel()for p in base.parameters()),'pair_parameters':params,'no_ID_or_private_forward_parameter':True});del base,probe
    out={}
    for arm in ['appearance_only','relative_geometry']:
        runs=[]
        for seed in range(3):
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);m=PairReader(arm=='relative_geometry').cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);batches=np.load(J/'logs'/f'balanced_loss_seed{seed}_batch_indices.npy');assert batches.shape==(1500,60);best=None;selected=None;log=[];start=time.monotonic();folder=E/'models'/arm/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
            for update,indices in enumerate(batches,1):
                ids=torch.tensor(indices.astype(np.int64),device='cuda');m.train();opt.zero_grad(set_to_none=True);o,d=m(**{k:z[ids]for k,z in dat.items()});logits=torch.cat([o[:,:,0],o[:,:,1],d],-1);losses=F.binary_cross_entropy_with_logits(logits,y[ids],reduction='none');parts=[]
                for g,cols in GROUPS.items():
                    mask=v[ids][:,cols];truth=y[ids][:,cols];w=truth*weights[g]['positive']+(1-truth)*weights[g]['negative'];parts.append((losses[:,cols]*mask*w).sum()/mask.sum())
                loss=torch.stack(parts).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                if update==1 or update%75==0:
                    pp={n:pred(m,d)for n,d in tune.items()};rank,sc=rank_predictions(pp,tune);path=folder/f'update{update:04d}.pt';torch.save({'model':m.state_dict(),'seed':seed,'update':update,'arm':arm,'protocol_sha256':sha(E/'configs/protocol.json'),'rank':rank},path);row={'update':update,'loss':float(loss.detach()),'rank':rank,'tune':sc,'checkpoint':str(path)};log.append(row)
                    if best is None or rank<best:best=rank;selected=row
                    save(E/'logs'/f'{arm}_seed{seed}.json',log)
            m.load_state_dict(torch.load(selected['checkpoint'],map_location='cuda',weights_only=False)['model']);preds={n:pred(m,ds[n])for n in ['old_tune','old_cal','new_dev']}
            for n,p in preds.items():np.save(E/'results'/f'{arm}_seed{seed}_{n}.npy',p)
            c=calibrate_joint({n:preds[n]for n in cal},cal);r={'seed':seed,'arm':arm,'selected_update':selected['update'],'checkpoint_path':selected['checkpoint'],'checkpoint_sha256':sha(selected['checkpoint']),'tune':selected['tune'],'calibration':c,'qualified':selected['tune']['gate']['passed']and c['eligible'],'updates':1500,'presentations':90000,'batch_indices_source':str(J/'logs'/f'balanced_loss_seed{seed}_batch_indices.npy'),'seconds':time.monotonic()-start};runs.append(r);save(E/'results'/f'{arm}_partial.json',runs);print(json.dumps({'arm':arm,'seed':seed,'selected':r['selected_update'],'qualified':r['qualified'],'tune_failures':r['tune']['gate']['failures'],'cal_failures':c['raw_gate']['failures']},ensure_ascii=False),flush=True);del m,opt
        out[arm]={'runs':runs,'qualified_count':sum(r['qualified']for r in runs),'raw_gate_failures_total':sum(r['tune']['gate']['failed_domain_predicates']+r['calibration']['raw_gate']['failed_domain_predicates']for r in runs),'mean_joint_tune_BCE':float(np.mean([r['tune']['joint_root_class_macro_BCE']for r in runs]))};save(E/'results'/f'{arm}.json',out[arm])
    a=out['appearance_only'];r=out['relative_geometry'];better=r['raw_gate_failures_total']<a['raw_gate_failures_total']or(r['raw_gate_failures_total']==a['raw_gate_failures_total']and r['mean_joint_tune_BCE']<=.8*a['mean_joint_tune_BCE']);out['collection_warranted']=r['qualified_count']>=2 and better;out['relational_dev_evidence_over_capacity_control']=better;out['tests_not_read']=True;save(E/'results/pair_training.json',out);print('COLLECTION_WARRANTED',out['collection_warranted'],flush=True)
if __name__=='__main__':main()
