from config import *
from losses import *
import fcntl
from runtime import load,tensor_hash
def main():
 assert not(E/'results/calibration.json').exists()
 for p,h in read(E/'configs/protocol_frozen.json')['sha256'].items():assert sha(p)==h,p
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and '/_private/'in str(args[0]):raise RuntimeError('No truth for calibration')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 rows,data=load_data();m,_,_=load(0,'canonical_matched');init=tensor_hash(m.state_dict());m.transition.requires_grad_(True);params=list(m.transition.parameters());records=[];used=set()
 for seed in [0,1]:
  sc=dict(np.load(E/'configs'/f'schedule_seed{seed}.npz'))
  for u in range(32):
   ix=list(sc['single'][u]);ix+=list(sc['multi'][u].ravel())if(u+1)%4==0 else[];assert all(rows[i]['split']=='train'for i in ix);used.update(ix)
   terms=batch_terms(m,data,sc,u);mixed=mixed_loss(terms);z=rawZ(terms);gm=torch.cat([g.ravel()for g in torch.autograd.grad(mixed,params,retain_graph=True)]);gz=torch.cat([g.ravel()for g in torch.autograd.grad(z,params)]);nm=float(gm.norm());nz=float(gz.norm());assert nz>0 and np.isfinite(nm+nz)
   records.append({'schedule_seed':seed,'update_index':u,'contains_multistep':(u+1)%4==0,'mixed_parameter_gradient_L2':nm,'rawZ_parameter_gradient_L2':nz,'cosine':float(torch.dot(gm,gz)/(gm.norm()*gz.norm())),'mixed_value':float(mixed.detach()),'rawZ_value':float(z.detach())})
 assert tensor_hash(m.state_dict())==init
 mm=float(np.median([r['mixed_parameter_gradient_L2']for r in records]));zm=float(np.median([r['rawZ_parameter_gradient_L2']for r in records]));alpha=mm/zm
 save(E/'results/calibration.json',{'finished_utc':now(),'alpha':alpha,'formula':'median mixed gradL2 / median rawZ gradL2','median_mixed_parameter_gradient':mm,'median_rawZ_parameter_gradient':zm,'median_scaledZ_parameter_gradient':alpha*zm,'uniqueABpairs':len(used),'batches':64,'multistep_batches':16,'C_examples_used':0,'D_examples_used':0,'optimizer_updates':0,'initial_model_hash':init,'final_model_hash':tensor_hash(m.state_dict()),'records':records})
 source=read(E/'configs/protocol_frozen.json');paths=list(source['sha256'])+[str(E/'results/calibration.json')];save(E/'configs/training_frozen.json',{'frozen_utc':now(),'alpha':alpha,'sha256':{p:sha(p)for p in paths}});print('CALIBRATED_ALPHA',alpha,'median_grad',mm,zm,'64ABbatches',flush=True)
if __name__=='__main__':main()
