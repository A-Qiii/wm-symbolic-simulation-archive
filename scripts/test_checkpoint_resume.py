from pathlib import Path
import json, os, signal, random, hashlib
import numpy as np
import torch
from wm_symbolic.checkpointing import capture_rng,restore_rng,atomic_save,learning_rate,PauseRequest
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes,batch_indices

P=Path('/home/robot/research/projects/wm_symbolic')
C=P/'runs/wm_pilot_protocol_v2_seed0/checkpoint_03000.pt'
c=torch.load(C,map_location='cpu',weights_only=False)
assert hashlib.sha256(C.read_bytes()).hexdigest()=='ba59dc834728d199f3f5158d0cb126b9614e9a5dc55f7244bdd0ed7a26f76774'
model=WorldModel(event_count=7);model.load_state_dict(c['model'],strict=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
optimizer.load_state_dict(c['optimizer'])
assert all(torch.equal(model.state_dict()[k],v) for k,v in c['model'].items())
assert all(int(s['step'])==3000 for s in optimizer.state.values())
assert optimizer.param_groups[0]['lr']==learning_rate(3000,6000)
data=Episodes(P,'train',normalizer=c['normalization'],event_count=7)
assert data.root_ids==c['identity']['train_roots'] and data.transitions==720
assert data.stats==c['normalization']
rng=np.random.default_rng(0);verified=0
for line in (C.parent/'batch_indices.jsonl').read_text().splitlines():
    record=json.loads(line)
    if record['update']>3000:break
    assert [batch_indices(data,rng).tolist() for _ in range(2)]==record['episodes']
    verified+=1
assert verified==3000
torch.manual_seed(19);random.seed(19);np.random.seed(19);rng=np.random.default_rng(19)
device='cuda';m=torch.nn.Linear(4,2).to(device);o=torch.optim.AdamW(m.parameters(),lr=.001)
def step():
    o.zero_grad();x=torch.randn(3,4,device=device)
    scalar=random.random()+np.random.random()+rng.random()
    loss=(m(x)*scalar).square().mean();loss.backward();o.step()
    return float(loss),{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
step();destination=P/'runs/checkpoint_recovery_test.pt'
atomic_save({'model':m.state_dict(),'optimizer':o.state_dict(),'rng':capture_rng(rng)},destination)
expected=step();saved=torch.load(destination,map_location='cpu',weights_only=False)
m.load_state_dict(saved['model']);o.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],rng)
actual=step();assert expected[0]==actual[0]
assert all(torch.equal(expected[1][k],actual[1][k]) for k in actual[1])
pause=PauseRequest();os.kill(os.getpid(),signal.SIGUSR1);assert pause.requested and pause.reason=='SIGUSR1'
result={'passed':True,'real_checkpoint_loaded':str(C),'model_exact_after_load':True,'optimizer_steps':3000,
 'normalization_reused':True,'previous_batch_updates_replayed_and_verified':verified,'next_update':3001,'next_lr':learning_rate(3001,6000),
 'atomic_checkpoint_roundtrip':True,'new_format_rng_and_optimizer_next_update_exact_in_cuda_toy_test':True,
 'safe_signal_handler_tested':True,'old_checkpoint_missing_rng':True,'test_scope':'Resume infrastructure, not research effectiveness'}
(P/'runs/checkpoint_resume_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
