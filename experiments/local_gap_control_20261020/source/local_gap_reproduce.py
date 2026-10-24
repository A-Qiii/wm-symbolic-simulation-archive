from common import *
import torch
guard();verify();checks=[]
for seed in [0,1]:
 p=S/f'models/coarse_seed{seed}_step1500.pt';q=OLD/f'models/geometry_seed{seed}_step1500.pt'
 a=torch.load(p,map_location='cpu',weights_only=False)['model'];b=torch.load(q,map_location='cpu',weights_only=False)['model'];assert a.keys()==b.keys()
 bad=[k for k in a if not torch.equal(a[k],b[k])]
 checks.append({'seed':seed,'tensors':len(a),'identical':not bad,'different_tensors':bad,'max_abs_difference':max(float((a[k]-b[k]).abs().max())for k in a),'current_checkpoint':str(p),'previous_checkpoint':str(q)})
save(S/'results/coarse_reproduction.json',{'utc':now(),'checks':checks,'model_tensors_compared_not_file_hashes':True});print(json.dumps(checks,indent=2))
