from config import *
import sys,torch,fcntl
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import Query,probs,feature_error
def main():
    f=read(E/'configs/choices_frozen.json')
    for p,h in f['sha256'].items():assert sha(p)==h,p
    def guard(event,args):
        if event=='open'and isinstance(args[0],(str,bytes))and'/_private/'in str(args[0]):raise RuntimeError('Evaluator inference cannot read truth')
    sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);a=dict(np.load(DIAG/'data/public/inputs.npz'));z=torch.tensor(a['z'],device='cuda');o=torch.tensor(a['objects'],device='cuda');s=torch.tensor(a['scene'],device='cuda');ts=read(DIAG/'data/public/trajectories.json');m,_,stats=load(0,'canonical_matched');frozen=tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')});out=[]
    with torch.inference_mode():
        for seed in[0,1]:
            q=Query().cuda();q.load_state_dict(torch.load(f['query_heads'][str(seed)]['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False)
            for arm in['mixed','latent']:
                name=f'{arm}_seed{seed}';m.load_state_dict(torch.load(f['selected'][name]['path'],weights_only=True,map_location='cuda')['state']);assert tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})==frozen
                for t in ts:
                    rr=t['steps'];b=torch.tensor([r['before']for r in rr],device='cuda');aft=torch.tensor([r['after']for r in rr],device='cuda');act=torch.tensor([r['action']for r in rr],device='cuda');one=m.transition(z[b],act);actual=z[aft]
                    for path,start in [('full4',0),('window1',1)]:
                        cur=z[b[start]:b[start]+1];rec=[]
                        for j in range(start,4):cur=m.transition(cur,act[j:j+1]);rec.append(cur[0])
                        values={'actual':actual[start:],'one_step':one[start:],'recursive':torch.stack(rec)};ll={k:q((v-stats['mean'])/stats['std'])for k,v in values.items()};pp={k:probs(l).cpu().numpy()for k,l in ll.items()};ll={k:l.cpu().numpy()for k,l in ll.items()};fe={k:feature_error(m.decode(v),o[aft[start:]],s[aft[start:]],(o[b[start:]],s[b[start:]])).cpu().numpy()for k,v in values.items()};lm={k:(v-z[aft[start:]]).square().mean((1,2)).cpu().numpy()for k,v in values.items()}
                        for j in range(4-start):out.append({k:t[k]for k in['group','root_id','color','branch']}|{'condition':name,'checkpoint_update':None,'path':path,'global_step':start+j+1,'local_depth':j+1,'skill':rr[start+j]['skill'],'before':int(b[start+j]),'after':int(aft[start+j]),**{k+'_probability':pp[k][j]for k in values},**{k+'_logits':ll[k][j]for k in values},**{k+'_feature_groups':fe[k][j]for k in values},**{k+'_raw_MSE':float(lm[k][j])for k in values}})
    save(E/'results/predictions.json',out);save(E/'results/evaluation_audit.json',{'rows':len(out),'private_truth_reads':0,'model_updates':0,'recursive_corrections_full4':0,'encoder_decoder_hash':frozen,'C_is_repeated_development':True,'D_used':False});print('EVALUATED',len(out),flush=True)
if __name__=='__main__':main()
