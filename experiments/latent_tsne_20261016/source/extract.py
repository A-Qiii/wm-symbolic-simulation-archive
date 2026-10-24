from common import *
import sys,fcntl,torch
# Load architecture only; never import a training entry point or any labels into forward computation.
sys.path.insert(0,str(OLD/'source'))
from model import Model
def load_model(kind):
 sel=read(OLD/'models'/f'{kind}_seed0'/'selected.json');assert sha(sel['path'])==sel['sha256'];norm=dict(np.load(OLD/'data/public/normalization.npz'));m=Model(norm,'learned').cuda().eval().requires_grad_(False);m.load_state_dict(torch.load(sel['path'],map_location='cuda',weights_only=True)['state']);return m,sel
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);states=read(E/'data/main_states.json');joint=read(E/'data/joint_pairs.json');data={k:dict(np.load(base/'public/inputs.npz'))for k,base in BASES.items()};zout={};evidence={}
 for kind in['interaction','ae']:
  m,sel=load_model(kind);o=np.stack([data[r['batch']]['objects'][r['state_index']]for r in states]);s=np.stack([data[r['batch']]['scene'][r['state_index']]for r in states])
  with torch.no_grad():z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda')).cpu().numpy()
  assert z.shape==(len(states),4,64);zout[kind]=z.reshape(len(states),256);evidence[kind]=sel
  if kind=='interaction':
   dd=data['D2'];ix=[r['after_index']for r in joint]
   with torch.no_grad():real=m.encode(torch.tensor(dd['objects'][ix],device='cuda'),torch.tensor(dd['scene'][ix],device='cuda')).cpu().numpy().reshape(len(ix),256)
   arrays={'actual':real}
   for arm in['canonical_control','canonical_matched']:
    sel2=read(REPAIR/'models'/f'{arm}_seed0'/'selected.json');cp=torch.load(sel2['path'],map_location='cpu',weights_only=True)['state'];assert all(torch.equal(v.cpu(),cp[k])for k,v in m.state_dict().items()if not k.startswith('transition.'));pred=dict(np.load(REPAIR/'results'/f'D2_{arm}_seed0.npz'));lookup={int(i):j for j,i in enumerate(pred['pair_indices'])};arrays[arm]=np.stack([pred['predicted_latent'][lookup[int(r['pair_id'][1:])]].reshape(256)for r in joint]);evidence[arm]=sel2
   np.savez_compressed(E/'data/joint_latents.npz',**arrays)
 np.savez_compressed(E/'data/main_latents.npz',**zout);save(E/'results/extraction.json',{'models':evidence,'main_states':len(states),'joint_pairs':len(joint),'latent_shape':[4,64],'flattened_dimensions':256,'query_outputs_used_in_projection':False,'model_updates':0,'new_physical_calls':0,'raw_DINO_features_used_as_projection_input':False,'repaired_encoder_exactly_equal_to_original':True,'device':torch.cuda.get_device_name(0)});print('CONTINUOUS_LATENTS_EXTRACTED',len(states),len(joint),flush=True)
if __name__=='__main__':main()
