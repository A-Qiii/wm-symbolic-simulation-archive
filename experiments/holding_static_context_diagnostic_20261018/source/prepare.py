from common import *
import shutil,torch
def main():
 assert not E.exists()and not R.exists();mp=B/'configs/delivery_manifest.json';assert sha(mp)=='9d8c3926e5d3a50ccb52287e47648f3fcab6caaf1fc9927bf4dc761fcaee0590'
 old=read(mp)
 for p,h in old['sha256'].items():assert sha(p)==h,p
 for n in ['configs','models','source','data','results','logs']:(E/n).mkdir(parents=True,exist_ok=True)
 (R/'figures').mkdir(parents=True);rows=read(B/'data/records.json');by_path={r['path']:r for r in rows if r['camera']=='static'};jobs=[];sync=[]
 for path,r in sorted(by_path.items()):
  d=B/'data/private'/r['root_id']/f's{r["sample"]:02d}';integrity=read(d/'integrity.json');static=read(d/'camera_static.json');wrist=read(d/'camera_gripper.json');assert static['tick']==wrist['tick']==integrity['tick'];assert integrity['four_same_tick_calls']and integrity['full_snapshot_and_contacts_equal']and integrity['RGB_depth_bitwise_equal']
  jobs.append({'path':path,'key':'rgb_static','RGB_sha256':r['RGB_sha256']});sync.append({'public_path':path,'tick':static['tick'],'same_tick':True})
 assert len(jobs)==264;save(E/'data/context_jobs.json',jobs);save(E/'results/context_synchronization.json',{'utc':now(),'states':264,'checks':sync,'model_receives':'RGB pixels only; no tick, root, sample, phase, action, pose, force or contact metadata'})
 oldinputs=read(B/'data/model_inputs.json');save(E/'data/query_context_map.json',[{'record_index':r['record_index'],'split':r['split'],'path':r['path']}for r in oldinputs])
 for rel in ['configs/schedule_seed0.npz','configs/schedule_seed1.npz','configs/visual_projection.npy']:shutil.copy2(B/rel,E/rel)
 initials=[]
 for seed in [0,1]:
  torch.manual_seed(261019210+seed);model=make();state=model.state_dict();torch.save({'state':state,'seed':seed,'architecture':'389-33-1'},E/f'models/initial_seed{seed}.pt');assert sum(p.numel()for p in model.parameters())==12904;initials.append({'seed':seed,'tensor_hash':tensor_hash(state),'tensor_shapes':{k:list(v.shape)for k,v in state.items()}})
 protocol={'utc':now(),'scope':'Final static-readout mechanism diagnostic on existing frozen36roots, already-seen C. Oracle boxes and synchronized multi-view scene context; not equal sensor budget to wrist-only.',
 'dataset':str(B),'queries':{'train':920,'dev':470},'samples':'same original eligibility/labels/root splits, no new collection or rendering; all failures/missingness unchanged',
 'features':'Existing261 pairRGB+relative2D unchanged, concatenate128 mean-patch DINOv2-small feature of full same-state staticRGB; bilinear224 align_cornersFalse, ImageNet normalization, same frozen384x128 data-independent projection. No new encoder weights/downloads. Both cameras share same context within state.',
 'model':'Linear389x33,SiLU,Linear33x1;12904 parameter scalars for both arms. Hidden33 chosen once, no sweep.',
 'arms':{'context_zero':'normalize389 using train query rows then replace last128 with exact0','context_present':'same389 normalized features, context retained'},'initializations':initials,
 'normalization':'train920query mean/popstd floor.05 for all389; first261 must equal original pair stats; shared by arms; no dev normalization fit',
 'training':{'seeds':[0,1],'torch_seeds':[261019210,261019211],'fits':4,'updates':600,'batch':32,'schedule':'original saved600x32 indices, identical within seed','optimizer':'AdamW','lr':.001,'weight_decay':.001,'BCE_posweight':4.75,'clip':1,'dtype':'float32','deterministic':True,'TF32':False,'selection':'final600 only','hard_negative_oversampling':False,'threshold':.5},
 'primary':{'target':'all original color==target_color queries','supported_negative':'original target, anchor closed_on_support and y==0','measures':'target P/R/BA/NLL and FP; supported-negative FP counts/FPR (no BA if no positives)','uncertainty':'original2000 paired complete12Croot bootstrap, pooled counts recomputed, percentile95CI, null undefined, count/FPR differences reported'},
 'secondary':['overall P/R/BA/NLL','per camera and color','train vs dev for all/target/supported-negative','C24/C34 both cameras and temporal neighbors; two-case success not sufficient'],
 'decision':'Clear primary gain requires both seeds targetBA difference95CI entirely positive AND supported-negative FPR difference95CI entirely negative. Otherwise report mixed/no clear gain and close static tuning. This is a diagnostic interpretation rule, not guide acceptance. No further tuning in this task regardless.',
 'limitations':['already-seen C exploratory, not confirmatory','context_zero new389fan-in initializations not equal previous261 model','static queries gain full-frame context; wrist queries gain a second synchronized camera','zero input reduces usable degrees of freedom despite identical tensor shapes','no explicit support-learning claim from classifier score alone','no deterministic planner integration'],
 'forbidden':['new data','threshold tuning','dev selection','label changes','best-seed choice','localization','SupCon','In','WM/transition fitting','closed loop','D sample/feature/prediction/score access'],
 'next_step':'Document needed evidence for controlled action-conditioned transition pilot only; do not collect or train it'}
 save(E/'configs/protocol.json',protocol);save(R/'FROZEN_PROTOCOL.json',protocol)
 refs=[mp,DINO/'model.safetensors',B/'data/records.json',B/'data/model_inputs.json',B/'configs/train_normalization.npz',B/'results/bootstrap_draws.npz']+[B/f'data/features/{s}.npz'for s in ['train','dev']]+[B/f'data/supervision/{s}_targets.json'for s in ['train','dev']];save(E/'configs/input_hashes.json',{str(p):sha(p)for p in refs})
 for src in Path(__file__).parent.glob('*.py'):shutil.copy2(src,E/'source'/src.name)
 files=[p for name in ['configs','models','source','data']for p in(E/name).rglob('*')if p.is_file()];save(E/'configs/pretraining_frozen.json',{'utc':now(),'before_features_and_fits':True,'sha256':{str(p):sha(p)for p in files}});print('FROZEN',sha(E/'configs/protocol.json'),'264 synchronized frames')
if __name__=='__main__':main()
