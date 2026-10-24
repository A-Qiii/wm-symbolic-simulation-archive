"""One prespecified executor diagnosis: change only requested colour on each C root."""
from pathlib import Path
import json,time,random
import numpy as np,torch,hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed
P=Path('/home/robot/research/projects/wm_symbolic')
out=P/'runs'/time.strftime('stage0_colour_binding_%Y%m%d_%H%M%S');out.mkdir()
annotations=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
evaluator=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
results=[]
for seed in range(3):
    env=CalvinAdapter(P,'C',seed,tabletop=True)
    try:
        env.reset()
        for _ in range(12):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
        root=env.evaluator_snapshot()
        np.savez_compressed(out/f'C_root{seed:03d}_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs'])
        for color in ['red','blue','pink']:
            env.reset(root);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            skill=f'lift_{color}_block_table'
            r=execute_fixed(env,policy,skill,annotations[skill][0],360,out/f'C_root{seed:03d}_{color}',evaluator)
            log=[json.loads(l) for l in (out/f'C_root{seed:03d}_{color}'/'trajectory_evaluator_only.jsonl').read_text().splitlines()]
            first=None
            for row in log:
                info=row['info'];robot=info['robot_info'];objects=info['scene_info']['movable_objects']
                touching={c[2] for c in robot['contacts']}
                touched=[name for name,value in objects.items() if value['uid'] in touching]
                if touched and robot['gripper_action']<0:
                    first={'step':row['step'],'touched_while_closing':touched};break
            r={k:v for k,v in r.items() if k!='executor'}
            r.update(root_seed=seed,first_closing_contact=first)
            results.append(r);print(json.dumps(r),flush=True)
            (out/'results.json').write_text(json.dumps(results,indent=2))
    finally:env.close()
print('Completed prespecified colour-only contrast:',out,flush=True)
