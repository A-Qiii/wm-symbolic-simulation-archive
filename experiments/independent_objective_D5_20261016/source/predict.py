from config import *
import sys,torch,fcntl
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import Query,probs,feature_error
def main():
 assert not(E/'results/predictions.json').exists()
 for p,h in read(E/'configs/before_collection_frozen.json')['sha256'].items():assert sha(p)==h,p
 protocol=read(E/'configs/protocol.json')
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and '/_private/'in str(args[0]):raise RuntimeError('Inference cannot read truth')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);d=E/'data/D5';data=dict(np.load(d/'public/inputs.npz'));z=torch.tensor(data['z'],device='cuda');o=torch.tensor(data['objects'],device='cuda');s=torch.tensor(data['scene'],device='cuda');pairs=read(d/'public/pairs.json');trajectories={}
 for r in pairs:trajectories.setdefault(r['root_id'],[]).append(r)
 m,_,stats=load(0,'canonical_matched');fixed=tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')});out=[];positions={}
 with torch.inference_mode():
  for seed in [0,1]:
   q=Query().cuda();q.load_state_dict(torch.load(protocol['query_heads'][str(seed)]['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);qh=tensor_hash(q.state_dict())
   for arm in ['mixed','latent']:
    name=f'{arm}_seed{seed}';cp=protocol['models'][name];m.load_state_dict(torch.load(cp['path'],weights_only=True,map_location='cuda')['state']);assert tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})==fixed;m.eval().requires_grad_(False);lat=[]
    for root,rr in trajectories.items():
     rr.sort(key=lambda r:r['slot']);assert [r['slot']for r in rr]==list(range(len(rr)));cur=z[rr[0]['before']:rr[0]['before']+1]
     for j,r in enumerate(rr):
      if j:assert rr[j-1]['after']==r['before']
      before=z[r['before']:r['before']+1];after=z[r['after']:r['after']+1];act=torch.tensor([r['action']],device='cuda');recbefore=cur.clone();one=m.transition(before,act);cur=m.transition(cur,act);values={'actual':after,'one_step':one,'recursive':cur};beforevalues={'actual':before,'one_step':before,'recursive':recbefore};record={k:r[k]for k in ['id','root_id','target','slot','skill','before','after']}|{'condition':name,'seed':seed,'global_step':r['slot']+1}
      for method,v in values.items():
       ll=q((v-stats['mean'])/stats['std']);prev=q((beforevalues[method]-stats['mean'])/stats['std']);record[method+'_probability']=probs(ll)[0].cpu().numpy();record[method+'_logits']=ll[0].cpu().numpy();record[method+'_before_probability']=probs(prev)[0].cpu().numpy();record[method+'_raw_MSE']=float((v-after).square().mean());record[method+'_feature_groups']=feature_error(m.decode(v),o[r['after']:r['after']+1],s[r['after']:r['after']+1],(o[r['before']:r['before']+1],s[r['before']:r['before']+1]))[0].cpu().numpy()
      out.append(record);lat.append({'pair':r['id'],'actual':after[0].cpu().numpy(),'one_step':one[0].cpu().numpy(),'recursive':cur[0].cpu().numpy()})
    np.savez_compressed(E/'results'/f'latents_{name}.npz',pair_ids=np.array([x['pair']for x in lat]),**{k:np.stack([x[k]for x in lat])for k in ['actual','one_step','recursive']});assert tensor_hash(q.state_dict())==qh;print('PREDICTED',name,len(lat),'observedboundaries',flush=True)
 save(E/'results/predictions.json',out);save(E/'results/prediction_audit.json',{'finished_utc':now(),'models':4,'observed_boundaries':len(pairs),'prediction_rows':len(out),'model_updates':0,'physics_calls':0,'private_truth_reads':0,'recursive_latent_corrections':0,'encoder_decoder_hash':fixed,'all_observed_success_and_failure_boundaries_included':True,'checkpoints_and_thresholds_unchanged':True});print('PREDICTION_DONE',len(out),flush=True)
if __name__=='__main__':main()
