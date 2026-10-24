from common import *
import shutil,collections,csv,torch

def main():
 assert not E.exists()
 for d in['configs','source','data/train','data/evaluation','models','results','logs']:(E/d).mkdir(parents=True)
 R.mkdir(parents=True);(R/'figures').mkdir();refs={}
 def reg(p):refs[str(p)]=sha(p);return p
 for p,h in read(reg(BASE/'configs/delivery_manifest.json'))['sha256'].items():assert sha(p)==h,p
 cfg=read(BASE/'configs/protocol.json');protocol={'utc':now(),'purpose':'Strict matched zero-context control plus read-only domain/error diagnosis after observing node/edge results','reference':str(BASE),'context_full':'reuse sealed full-edge4trained heads and their scores; no reference refit','new_fits':4,'heads':HEADS,'seeds':[0,1],'updates_each':600,'batch_states':32,'architecture':'identical207->32->SiLU->1,6689params','initialization':'exact copied full-edge initialization tensors perhead/seed, not just same seed','difference':'after original normalization, set channels74:207 to exact0 during both training and inference; keep channels0:74 bit-identical','zeroed_channels':'endpoint visual128 and image-relative geometry5; constant zero contains no endpoint presence or source metadata','schedule':'exact full-edge24old+8official600x32 schedule','optimizer':'identical AdamW lr.001 wd.001 clip1; same masked positive-weighted BCE','positive_weights':cfg['positive_weights'],'old_pool_states':cfg['old_pool_states'],'normalization':'copy full-edge train-only normalization unchanged','no_tuning':'all validation cohorts already viewed; no new independent evaluation,threshold.5,no seed/weight/ROI sweep','scope':'fixed-scene contextual region probe, not generic instance graph; full-image DINO node tokens already contain context','new_data':False,'T_or_DINO_training':False,'SupCon':'remains paused','model_adoption':False}
 save(E/'configs/protocol.json',protocol);tests=[]
 for head in HEADS:
  d=np.load(reg(BASE/f'data/train/{head}.npz'));full=d['edge'].copy();zero=full.copy();zero[:,:,74:]=0
  assert np.array_equal(zero[:,:,:74],full[:,:,:74])and np.count_nonzero(zero[:,:,74:])==0
  np.savez_compressed(E/f'data/train/{head}.npz',x=zero,y=d['y'],v=d['v'],indices=d['indices'],state_ids=d['state_ids'],root_ids=d['root_ids'])
  shutil.copy2(reg(BASE/f'configs/norm_{head}_edge.npz'),E/f'configs/norm_{head}.npz')
  for seed in(0,1):
   shutil.copy2(reg(BASE/f'configs/schedule_{head}_seed{seed}.npz'),E/f'configs/schedule_{head}_seed{seed}.npz')
   src=reg(BASE/f'models/initial_{head}_edge_seed{seed}.pt');dst=E/f'models/initial_{head}_seed{seed}.pt';shutil.copy2(src,dst);initial=torch.load(dst,weights_only=True,map_location='cpu');q=make();q.load_state_dict(initial['state']);assert tensor_hash(q.state_dict())==initial['initial_hash'];assert sha(src)==sha(dst)
   ref=reg(BASE/f'models/{head}_edge_seed{seed}_update0600.pt');shutil.copy2(ref,E/f'models/reference_full_{head}_seed{seed}.pt')
   tests.append({'head':head,'seed':seed,'initial_tensor_hash_equal':True,'training_object_channels_equal':True,'all_extra_channels_zero_after_normalization':True,'parameters':sum(p.numel()for p in q.parameters())})
 for p in(BASE/'data/evaluation').glob('*.npz'):shutil.copy2(reg(p),E/'data/evaluation'/p.name)
 save(E/'results/control_preflight.json',{'utc':now(),'passed':True,'checks':tests,'fits_so_far':0});save(E/'configs/source_references.json',{'sha256':refs})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 print('MATCHED_ZERO_CONTROL_PREPARED',flush=True)
if __name__=='__main__':main()

