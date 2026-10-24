from config import *
import sys,torch,fcntl
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import Query,probs
def main():
 verify_frozen();assert not(E/'results/predictions.json').exists();cfg=read(E/'configs/protocol.json');norm=dict(np.load(OBJ/'configs/object_normalization.npz'))
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and '/_private/'in str(args[0]):raise RuntimeError('Prediction cannot read physical truth')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);data=np.load(E/'data/combo/public/inputs.npz');z=torch.tensor(data['z'],device='cuda');pairs=read(E/'data/combo/public/pairs.json');trajectories={}
 for p in pairs:trajectories.setdefault((p['root_id'],p['branch']),[]).append(p)
 m,_,stats=load(0,'canonical_matched');fixed=tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')});out=[];nm=torch.tensor(norm['mean'],device='cuda');ns=torch.tensor(norm['std'],device='cuda')
 with torch.inference_mode():
  for seed in [0,1]:
   q=Query().cuda().eval().requires_grad_(False);q.load_state_dict(torch.load(cfg['query_heads'][str(seed)]['original']['path'],map_location='cuda',weights_only=True)['state']);qh=tensor_hash(q.state_dict());iq={}
   for arm in ['object_control','object_coverage']:
    iq[arm]=torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.SiLU(),torch.nn.Linear(32,1)).cuda().eval().requires_grad_(False);iq[arm].load_state_dict(torch.load(cfg['query_heads'][str(seed)][arm]['path'],map_location='cuda',weights_only=True)['state'])
   for wm in ['mixed','latent']:
    name=f'{wm}_seed{seed}';m.load_state_dict(torch.load(cfg['models'][name]['path'],map_location='cuda',weights_only=True)['state']);assert tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})==fixed;m.eval().requires_grad_(False);lat=[]
    for(root,branch),rr in trajectories.items():
     rr=sorted(rr,key=lambda r:r['slot']);assert [r['slot']for r in rr]==list(range(len(rr)));cur=z[rr[0]['before']:rr[0]['before']+1]
     for j,r in enumerate(rr):
      if j:assert rr[j-1]['after']==r['before']
      before=z[r['before']:r['before']+1];after=z[r['after']:r['after']+1];act=torch.tensor([r['action']],device='cuda');one=m.transition(before,act);cur=m.transition(cur,act);values={'actual':after,'one_step':one,'recursive':cur};prob={p:probs(q((v-stats['mean'])/stats['std']))[0].cpu().numpy()for p,v in values.items()};meta={k:r[k]for k in ['id','root_id','environment','target','other_target','branch','slot','skill','before','after']}|{'seed':seed,'world_model':wm}
      for arm in ARMS:
       row=meta|{'query_arm':arm}
       for path,v in values.items():
        pp=prob[path].copy()
        if arm!='original':pp[3:6]=iq[arm]((v[:,:3]-nm)/ns).squeeze(-1).sigmoid()[0].cpu().numpy()
        row[path+'_probability']=pp
       out.append(row)
      lat.append({'pair':r['id'],**{k:v[0].cpu().numpy()for k,v in values.items()}})
    np.savez_compressed(E/'results'/f'latents_{name}.npz',pair_ids=np.array([r['pair']for r in lat]),**{k:np.stack([r[k]for r in lat])for k in METHODS});assert tensor_hash(q.state_dict())==qh;print('IDENTITY_PREDICTED',name,len(lat),'boundaries',flush=True)
 save(E/'results/predictions.json',out);save(E/'results/prediction_audit.json',{'utc':now(),'model_updates':0,'physics_calls':0,'private_truth_reads':0,'rows':len(out),'observed_transitions':len(pairs),'encoder_decoder_hash':fixed,'original_Holding_drawer_reused':True,'trueHolding_mask':False,'threshold_or_model_selection':False,'recursive_corrections':0});print('PREDICTION_DONE',len(out),flush=True)
if __name__=='__main__':main()
