from common import *
from net import *
import baseline_net,shutil

def main():
 assert not E.exists()
 for sub in ['configs','source','models','data/predictions','data/baseline','data/BCE','data/evaluation','results','logs']:(E/sub).mkdir(parents=True,exist_ok=True)
 R.mkdir();(R/'figures').mkdir()
 inputs=[BASE/'configs/delivery_manifest.json',BASE/'configs/training_frozen.json',BASE/'configs/training_runtime.json',BASE/'configs/normalization.npz',BASE/'data/train/train.npz',BASE/'source/net.py',PRBASE/'configs/delivery_manifest.json',PRBASE/'data/ABC_labels_metadata.json',PRBASE/'results/summary.json',PRBASE/'results/fixed_threshold_points.json',PRBASE/'results/slot_and_GT_overlap.json',BCEBASE/'configs/delivery_manifest.json',BCEBASE/'results/evaluation_summary.json',BCEBASE/'results/fixed_threshold_points.json']
 for seed in [0,1]:
  inputs.extend([BASE/f'models/initial_seed{seed}.pt',BASE/f'models/final_seed{seed}.pt',BASE/f'configs/schedule_seed{seed}.npy',BASE/f'results/train_seed{seed}.json',BCEBASE/f'models/final_seed{seed}.pt',BCEBASE/f'results/train_seed{seed}.json'])
  for split in ['train','dev']:
   for cam in ['all','static','gripper']:
    for cat in ['all']+CATS:
     inputs.extend([PRBASE/f'data/pr_seed{seed}_{split}_{cam}_{cat}.npz',BCEBASE/f'data/evaluation/pr_seed{seed}_{split}_{cam}_{cat}.npz'])
 inputs.extend(BASE/f'data/features/{s}.npz'for s in ['train','dev']);opened=install_guard(inputs)
 seals={p:read(p/'configs/delivery_manifest.json')['sha256']for p in [BASE,PRBASE,BCEBASE]};hashes={}
 for p in inputs:
  h=sha(p);hashes[str(p)]=h
  if p.name!='delivery_manifest.json':assert h==next(v for root,v in seals.items()if p.is_relative_to(root))[str(p)],p
 local=Path(__file__).parent;old=(BASE/'source/net.py').read_text();assert old==(local/'baseline_net.py').read_text()
 expr='weight=torch.where(y>0,.25*(1-pr).square(),.75*pr.square())';replacement='weight=torch.where(torch.arange(3,device=y.device)[None,None,:]==0,torch.where(y>0,.25,.75),torch.where(y>0,.25*(1-pr).square(),.75*pr.square()))';assert old.count(expr)==1 and old.replace(expr,replacement)==(local/'net.py').read_text()
 runtime=read(BASE/'configs/training_runtime.json');assert runtime['torch']==torch.__version__ and runtime['numpy']==np.__version__;assert runtime['trainable_parameters']==sum(p.numel()for p in make().parameters())==25615
 protocol={'utc':now(),'hypothesis':'Only front slot0 uses gamma0; finger slots1/2 retain gamma2. Shared hidden layer couples all outputs, so this cannot be predicted by splicing prior score tables.','exploratory':'Hypothesis proposed after observing C. C remains development evidence, never new independent validation.','changed_factor':{'gamma_by_slot':[0,2,2],'alpha_positive':.25,'alpha_negative':.75,'normalization':'same per-image sum / max(all positive centers,1), then batch mean','source_change':'one weight expression only'},'unchanged':{'head':'shared Linear384->64 SiLU Linear64->15, 25615 parameters','data':'same saved train.npz and frozen DINO train/dev features','normalization':'same saved mean/std','regression':'5 SmoothL1 + 2 GIoU, same masks, centers and targets','seeds':[0,1],'updates_each':1200,'batch':16,'initialization_and_schedule':'original saved files reused byte for byte','optimizer':'same AdamW lr .001 wd .0001 defaults; clip 1','determinism':'same CUBLAS :4096:8, no TF32/AMP/augmentation','decode':'full normal candidates, class NMS IoU>0.5, match IoU>=0.5, all valid GT denominator','work_point':.5},'candidate_selection':{'frozen_before_hybrid_fit':True,'comparator':'original focal, seed-paired','rule':'A candidate must, in BOTH seeds, have C micro overall/front/finger AP50 >= original focal, C micro precision@0.5 >= original focal and recall@0.5 >= original focal, and strictly increase front AP50. No averaging away a failed seed or endpoint; epsilon 1e-12 only for float equality. Otherwise retain original focal.','eligible_variants':['full_BCE','mixed'],'if_both_pass':'prefer mixed because it retains the original finger objective; fixed priority, not best score','cameras':'report all camera strata as diagnostics, not an additional post-hoc selection test','training_scores':'diagnostic only','seed_policy':'select a loss configuration, keep both seeds; do not deploy or select best seed','not_production_acceptance':'passing is only eligibility for new-root verification, not a reliability or research-success gate'},'evaluation':{'splits':['train','dev'],'D_reads':0,'thresholds':[.5,.3,.1],'threshold_selection':False,'new_data_collection':False,'full_AP':'all-points precision-envelope AP50, micro all classes and class/camera strata; no oracle','prior_variants':'reuse sealed ABC curves/results without new fits or full inference'},'diagnostics':{'initial_fixed_batches':'same root_balanced and rare_slot as prior; compare all three objectives at exactly same weights, zero updates','shared_layer':'compare per-class gradients on shared layer; verify mixed and focal finger direct gradients equal before global clip, while front/shared gradients differ; compare final parameter groups; changes are not proof of isolated final AP causality','rare_exposures':'original schedule [14250,37425,825] per seed'},'stop':'This is the last loss-selection contrast. Exactly two new fits; no more gamma, alpha, mixture-weight or C-driven loss search. Prepare new-root/threshold protocol only, do not acquire or score new tests.','allowlisted_old_inputs':[str(p)for p in inputs]}
 save(E/'configs/protocol.json',protocol);save(E/'configs/input_frozen.json',{'utc':now(),'sha256':hashes})
 z=np.load(BASE/'data/train/train.npz',allow_pickle=False);y,v,b=z['y'],z['v'],z['b'];assert np.array_equal(z['image_ids'],np.arange(256));assert y.shape==(256,1024,3)and np.all(y*(1-v[:,None,:])==0);assert y.sum((0,1)).tolist()==[190,499,11]
 rare=np.where(y[:,:,2].sum(1)>0)[0].tolist();probes={'root_balanced':[16*r+r%2 for r in range(16)],'rare_slot':rare+[i for i in range(256)if i not in rare][:5]};save(E/'configs/probe_batches.json',probes);coverage=[]
 for seed in [0,1]:
  schedule=np.load(BASE/f'configs/schedule_seed{seed}.npy');assert schedule.shape==(1200,16)and np.all(np.bincount(schedule.ravel(),minlength=256)==75)
  for a,d in [(BASE/f'configs/schedule_seed{seed}.npy',E/f'configs/schedule_seed{seed}.npy'),(BASE/f'models/initial_seed{seed}.pt',E/f'models/initial_seed{seed}.pt')]:shutil.copy2(a,d)
  initial=torch.load(E/f'models/initial_seed{seed}.pt',weights_only=True,map_location='cpu');assert tensor_hash(initial['state'])==initial['initial_hash'];coverage.append({'seed':seed,'positive_exposures_by_slot':y.sum(1)[schedule].sum((0,1)).tolist(),'batches_containing_collision_positive':int((y[:,:,2].sum(1)[schedule].sum(1)>0).sum()),'initial_hash':initial['initial_hash']})
 torch.manual_seed(261018);raw=torch.randn(16,1024,15,requires_grad=True);yy,vv,bb=map(torch.tensor,(y[:16],v[:16],b[:16]));lm,pm=loss(raw,yy,vv,bb);lf,pf=baseline_net.loss(raw,yy,vv,bb);assert torch.equal(pm[1:],pf[1:]);log,_=unpack(raw);prob=log.sigmoid();bbce=F.binary_cross_entropy_with_logits(log,yy,reduction='none');w=torch.where(yy>0,.25*(1-prob).square(),.75*prob.square());w[:,:,0]=torch.where(yy[:,:,0]>0,.25,.75);manual=((bbce*w*vv[:,None,:]).sum((1,2))/yy.sum((1,2)).clamp(min=1)).mean();assert torch.allclose(lm,manual+5*pm[1]+2*pm[2],atol=1e-6,rtol=1e-6)
 gm=torch.autograd.grad(lm,raw,retain_graph=True)[0].reshape(16,1024,3,5);gf=torch.autograd.grad(lf,raw)[0].reshape(16,1024,3,5);assert torch.equal(gm[:,:,1:,:],gf[:,:,1:,:]);assert torch.equal(gm[:,:,:,1:],gf[:,:,:,1:])
 zero=torch.randn(1,1024,15,requires_grad=True);lz,_=loss(zero,torch.zeros(1,1024,3),torch.zeros(1,3),torch.zeros(1,1024,3,4));lz.backward();assert lz.item()==0 and torch.count_nonzero(zero.grad)==0
 copies={E/'configs/normalization.npz':BASE/'configs/normalization.npz',E/'data/ABC_labels_metadata.json':PRBASE/'data/ABC_labels_metadata.json'}
 for name in ['summary.json','fixed_threshold_points.json','slot_and_GT_overlap.json']:copies[E/'data/baseline'/name]=PRBASE/'results'/name
 for dest,src in [('summary.json','evaluation_summary.json'),('fixed_threshold_points.json','fixed_threshold_points.json')]:copies[E/'data/BCE'/dest]=BCEBASE/'results'/src
 for seed in [0,1]:
  copies[E/f'data/baseline/train_seed{seed}.json']=BASE/f'results/train_seed{seed}.json';copies[E/f'data/BCE/train_seed{seed}.json']=BCEBASE/f'results/train_seed{seed}.json'
  for split in ['train','dev']:
   for cam in ['all','static','gripper']:
    for cat in ['all']+CATS:
     n=f'pr_seed{seed}_{split}_{cam}_{cat}.npz';copies[E/'data/baseline'/n]=PRBASE/'data'/n;copies[E/'data/BCE'/n]=BCEBASE/'data/evaluation'/n
 for dest,src in copies.items():shutil.copy2(src,dest)
 save(E/'configs/copied_input_map.json',{str(d):str(s)for d,s in copies.items()})
 save(E/'results/preflight.json',{'utc':now(),'passed':True,'source_diff_exactly_one_weight_expression':True,'regression_components_bitwise_equal':True,'finger_output_gradients_bitwise_equal':True,'all_regression_output_gradients_bitwise_equal':True,'unknown_zero_gradient':True,'old_sealed_selected_inputs_verified':len(inputs),'coverage':coverage,'old_files_opened':sorted(opened),'D_reads':0,'optimizer_updates':0})
 for f in local.glob('*.py'):shutil.copy2(f,E/'source'/f.name)
 files=[p for sub in ['configs','source','models']for p in(E/sub).glob('*')if p.is_file()];save(E/'configs/pre_run_frozen.json',{'utc':now(),'before_first_new_optimizer_update':True,'sha256':{str(p):sha(p)for p in files}});print('MIXED_PROTOCOL_FROZEN',len(inputs),flush=True)
if __name__=='__main__':main()
