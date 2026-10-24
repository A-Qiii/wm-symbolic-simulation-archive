"""Fixed four-skill object reference, using the same frozen executor on C."""
from pathlib import Path
import time,json,random
import numpy as np,torch,hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed
P=Path('/home/robot/research/projects/wm_symbolic')
out=P/'runs'/time.strftime('stage0_container_reference_%Y%m%d_%H%M%S');out.mkdir()
ann=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
results=[]
for seed in [0,1,2]:
    env=CalvinAdapter(P,'C',seed,tabletop=True)
    try:
        env.reset()
        for _ in range(12):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        for step,skill in enumerate(['open_drawer','lift_pink_block_table','place_in_drawer','close_drawer']):
            r=execute_fixed(env,policy,skill,ann[skill][0],360,out/f'C_root{seed:03d}_{step}_{skill}',checker)
            results.append({k:v for k,v in r.items() if k!='executor'}|{'seed':seed,'step':step})
            (out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(results[-1]),flush=True)
    finally:env.close()
print(out,flush=True)
