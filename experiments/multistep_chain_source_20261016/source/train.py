from config import *
import sys,torch,fcntl,argparse
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import feature_error,Query

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['old_chains','new_chains'],required=True);ap.add_argument('--seed',type=int,choices=[0,1],required=True);a=ap.parse_args()
    for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
    protocol=read(E/'configs/protocol.json');qsel=protocol['query_heads'][str(a.seed)]
    allowed={str(p)for p in [PACK/'inputs.npz',PACK/'pairs.json',E/'configs'/f'schedule_seed{a.seed}.npz',Path(qsel['path']),REP/'data/public/normalization.npz',REP/'models/canonical_matched_seed0/selected.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt']}
    def guard(event,args):
        if event=='open'and isinstance(args[0],(str,bytes)):
            p=Path(args[0]).resolve()
            if p.is_relative_to(P/'experiments')and p.suffix in ['.json','.npz','.pt']and str(p)not in allowed and not p.is_relative_to(E/'models')and not p.is_relative_to(E/'results'):raise RuntimeError('Transition trainer forbidden data '+str(p))
    sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=True
    rows=read(PACK/'pairs.json');data=dict(np.load(PACK/'inputs.npz'));o=torch.tensor(data['objects'],device='cuda');s=torch.tensor(data['scene'],device='cuda');z=torch.tensor(data['z'],device='cuda')
    b=torch.tensor([r['before']for r in rows],device='cuda');after=torch.tensor([r['after']for r in rows],device='cuda');act=torch.tensor([r['action']for r in rows],device='cuda');sc=dict(np.load(E/'configs'/f'schedule_seed{a.seed}.npz'))
    m,_,stats=load(0,'canonical_matched');q=Query().cuda();q.load_state_dict(torch.load(qsel['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);m.transition.requires_grad_(True)
    def fh():return tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})
    init=tensor_hash(m.state_dict());frozen=fh();qh=tensor_hash(q.state_dict());opt=torch.optim.AdamW(m.transition.parameters(),lr=.0001,weight_decay=.0001)
    folder=E/'models'/f'{a.arm}_seed{a.seed}';folder.mkdir();logs=[];best=float('inf');selected=None;dev=[i for i,r in enumerate(rows)if r['split']in ['dev','dev_select']]
    def checkpoint(u,trainloss):
        nonlocal best,selected
        with torch.no_grad():pp=m.transition(z[b[dev]],act[dev]);errs=feature_error(m.decode(pp),o[after[dev]],s[after[dev]],(o[b[dev]],s[b[dev]])).mean(1).cpu().numpy()
        vals={root:float(errs[[j for j,i in enumerate(dev)if rows[i]['root_id']==root]].mean())for root in sorted({rows[i]['root_id']for i in dev})}
        old=[v for k,v in vals.items()if not k.startswith('C_intermediate_v3_')];new=[v for k,v in vals.items()if k.startswith('C_intermediate_v3_')];assert len(old)==37 and len(new)==6
        score=.5*(float(np.mean(old))+float(np.mean(new)));assert fh()==frozen and tensor_hash(q.state_dict())==qh
        path=folder/f'update{u:04d}.pt';torch.save({'state':m.state_dict(),'update':u,'C_score':score,'frozen_hash':frozen},path)
        logs.append({'update':u,'score':score,'old_C_mean':float(np.mean(old)),'new_C_mean':float(np.mean(new)),'train_loss':trainloss});print(a.arm,a.seed,u,score,flush=True)
        if score<best:best=score;selected=path
    checkpoint(0,None)
    for u in range(1000):
        ix=sc['single'][u];pred=m.transition(z[b[ix]],act[ix]);loss=feature_error(m.decode(pred),o[after[ix]],s[after[ix]],(o[b[ix]],s[b[ix]])).mean()+.25*(pred-z[after[ix]]).square().mean()
        if (u+1)%4==0:
            seq=sc[a.arm][u];assert (seq>=0).all();first=seq[:,0];cur=m.transition(z[b[first]],act[first])
            for depth in [1,2]:
                ni=seq[:,depth];cur=m.transition(cur,act[ni]);loss+=.25*(feature_error(m.decode(cur),o[after[ni]],s[after[ni]],(o[b[ni]],s[b[ni]])).mean()+.25*(cur-z[after[ni]]).square().mean())
        opt.zero_grad();assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in m.named_parameters()if not n.startswith('transition.'));assert all(p.grad is None for p in q.parameters());torch.nn.utils.clip_grad_norm_(m.transition.parameters(),5);opt.step()
        if (u+1)%250==0:checkpoint(u+1,float(loss.detach()))
    save(folder/'selected.json',{'path':str(selected),'sha256':sha(selected),'C_score':best})
    save(E/'results'/f'train_{a.arm}_seed{a.seed}.json',{'updates':1000,'single_draws':64000,'multistep_draws':12000,'initial_model_hash':init,'frozen_encoder_decoder_hash':frozen,'final_frozen_hash':fh(),'query_hash':qh,'final_query_hash':tensor_hash(q.state_dict()),'query_path':qsel['path'],'log':logs,'only_transition_optimized':True,'validation_or_test_access':False})
    print('TRAINING_DONE',a.arm,a.seed,flush=True)

if __name__=='__main__':main()
