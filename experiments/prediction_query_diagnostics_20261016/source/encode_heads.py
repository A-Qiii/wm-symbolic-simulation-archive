"""Prepare disjoint cached true-state packs; no optimizer or D/D2 access."""
from config import *
import torch,sys,fcntl
sys.path.insert(0,str(REP/'source'))
from runtime import load,normalize
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);sel=read(E/'configs/head_selection.json');states=read(REP/'data/public/states.json');labels=read(REP/'data/_private/labels.json');d=dict(np.load(REP/'data/public/inputs.npz'));m,q,stats=load(0,'canonical_matched');ix=sorted(set(sum(sel['pools'].values(),[])+sel['C_selection_indices']+sel['C_validation_indices']));assert all(states[i]['split']in['train','dev']and states[i]['root_id'].startswith(('A_','B_','C_'))for i in ix)
 with torch.inference_mode():z=m.encode(torch.tensor(d['objects'][ix],device='cuda'),torch.tensor(d['scene'][ix],device='cuda'));zn=normalize(z,stats).cpu().numpy()
 lookup={i:j for j,i in enumerate(ix)}
 for name,ids in list(sel['pools'].items())+[('C_selection',sel['C_selection_indices']),('C_validation',sel['C_validation_indices'])]:
  directory='head_train'if name!='C_validation'else'evaluator';np.savez_compressed(E/'data'/directory/f'{name}.npz',z=zn[[lookup[i]for i in ids]],y=np.array([labels[str(i)]['y']for i in ids],np.float32),v=np.array([labels[str(i)]['v']for i in ids],np.float32),state_ids=np.array(ids),root_ids=np.array([states[i]['root_id']for i in ids]))
 save(E/'results/head_encoding.json',{'fixed_world_model_sha256':sha(REP/'models/canonical_matched_seed0/update1000.pt'),'fixed_original_query_sha256':sha(REP/'models/query_seed0.pt'),'states_encoded':len(ix),'representation_updates':0,'D_or_D2_states':0,'normalization_changed':False});print('HEAD_TRUE_STATE_PACKS_READY',len(ix),flush=True)
if __name__=='__main__':main()
