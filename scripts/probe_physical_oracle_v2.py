from pathlib import Path
import json,argparse,time,hashlib
import numpy as np
import pybullet as p
import imageio.v2 as imageio
import hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2 as PhysicalOracle

P=Path('/home/robot/research/projects/wm_symbolic');BASE=P/'runs/physical_oracle_v1';OUT=P/'runs/physical_oracle_v2'
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--root',default='C_development_00');pa.add_argument('--skills',nargs='+',default=['open_drawer','lift_red_block_table','place_in_drawer','close_drawer']);pa.add_argument('--name',required=True);args=pa.parse_args()
    roots=json.loads((BASE/'roots_manifest.json').read_text());row=next(r for r in roots if r['root_id']==args.root)
    # All v1 roots are now development/regression; fresh v2 roots remain unused.
    out=OUT/'development'/args.name;out.mkdir(parents=True,exist_ok=False)
    env=CalvinAdapter(P,row['environment'],row['seed'],tabletop=True)
    oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
    results=[]
    try:
      data=np.load(BASE/'roots'/args.root/'reset_evaluator_only.npz');env.reset({'environment':row['environment'],**dict(data)})
      for i,skill in enumerate(args.skills):
        d=out/f'{i:02d}_{skill}';d.mkdir();obs=env.observe();start=env.env.get_info();save_npz(d/'initial_public.npz',obs)
        videos={k:imageio.get_writer(str(d/(k+'.mp4')),fps=30,macro_block_size=1,codec='libx264',quality=7) for k in obs['rgb_obs']}
        for k,w in videos.items():w.append_data(obs['rgb_obs'][k])
        controls=[];private=[]
        def record(obs,act):
          controls.append(act)
          for k,w in videos.items():w.append_data(obs['rgb_obs'][k])
          private.append({'robot_obs':obs['robot_obs'].tolist(),'objects':{o.name:o.get_state().tolist() for o in env.env.scene.movable_objects},'contacts':[list(c) for c in p.getContactPoints(bodyA=env.env.robot.robot_uid,physicsClientId=env.env.cid) if c[9]>.01], 'scene_obs':env.env.scene.get_obs().tolist()})
        controller=PhysicalOracle(env,record)
        try:r=controller.run(skill)
        finally:
          for w in videos.values():w.close()
          controller.close()
        r.update(skill=skill,official_success=skill in oracle.get_task_info_for_set(start,env.env.get_info(),{skill}),root=args.root,controller=controller.VERSION)
        (d/'result.json').write_text(json.dumps(r,indent=2));(d/'controls.json').write_text(json.dumps(controls));(d/'trace_private.json').write_text(json.dumps(private));save_npz(d/'final_public.npz',env.observe())
        results.append(r);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
        if r['status']!='success':break
    finally:env.close()
if __name__=='__main__':main()
