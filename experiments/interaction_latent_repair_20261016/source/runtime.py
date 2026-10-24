from common import *
from model import Model,Query
import torch
ARMS=['canonical_control','canonical_matched']
def normalize(z,stats):return (z-stats['mean'])/stats['std']
def tensor_hash(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load(seed,arm=None,device='cuda',initial=False):
 norm=dict(np.load(E/'data/public/normalization.npz'));m=Model(norm,'learned').to(device)
 if initial or arm is None:path=E/'models'/f'initial_seed{seed}.pt'
 else:
  sel=read(E/'models'/f'{arm}_seed{seed}'/'selected.json');path=Path(sel['path']);assert sha(path)==sel['sha256']
 cp=torch.load(path,map_location=device,weights_only=True);m.load_state_dict(cp['state']);m.eval().requires_grad_(False)
 qc=torch.load(E/'models'/f'query_seed{seed}.pt',map_location=device,weights_only=True);q=Query().to(device);q.load_state_dict(qc['state']);q.eval().requires_grad_(False)
 return m,q,qc['normalization']
