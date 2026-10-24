"""Freeze physical reset states before developing a separate GT-assisted executor."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import pybullet as p
from PIL import Image
from wm_symbolic.env_adapter import CalvinAdapter, save_npz

P=Path('/home/robot/research/projects/wm_symbolic')
OUT=P/'runs/physical_oracle_v1'
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    protocol={'version':'physical_oracle_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'authorization':'Explicit user: establish reliable control endpoint; simulator GT and physical scripted/IK trajectories allowed.',
      'scope':'New independent executor. Main 4215 transitions/HULC unchanged. No formal collection/training.',
      'boundary':'GT confined to controller and private evaluator. Public observation allowlist unchanged. No high-level planner or WM receives GT.',
      'physics':'Unmodified CALVIN URDF/dynamics/contact, standard motor actions. No object teleport/reset, collision disable, artificial attachment or external object forces during skills.',
      'roots':'9 development and18 independent acceptance, native A/B/C; no D. Deterministic physically separated tabletop placements; no outcome screening.',
      'skills':['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer'],
      'chain':'open_drawer -> lift designated color -> place_in_drawer -> close_drawer',
      'acceptance_chain_colors':'Each root has predeclared color; 2 roots/color/environment. Stop chain on first failed skill, retain all failures.',
      'negative_cases':['place_in_drawer empty gripper','close_drawer already closed','lift then place_in_drawer while drawer remains closed'],
      'acceptance':'Report successes and correct precondition refusals separately. Target all feasible predeclared chains, no claim universal reliability. Hold code fixed before acceptance; no tune on acceptance.',
      'limits':{'max_controls_per_skill':900,'max_attempts_per_skill':2,'completion_stability_controls':12},
      'audits':['controls','attempts/retries','official task completion','final goal validity','contacts/penetration','unrequested block/device changes','dual RGB cameras','reset hash'],
      'main_manifest_sha256':sha(P/'data/raw/transitions.jsonl') if (P/'data/raw/transitions.jsonl').exists() else None}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for split,n,seedbase in [('development',3,71000),('acceptance',6,81000)]:
      for ei,envname in enumerate('ABC'):
        env=CalvinAdapter(P,envname,seedbase+100*ei,tabletop=True)
        try:
          lo,hi=np.asarray(env.env.scene.surfaces['table'])
          for i in range(n):
            seed=seedbase+100*ei+i; rng=np.random.RandomState(seed)
            names=rng.permutation(['block_red','block_blue','block_pink'])
            layout={name:[float(lo[0]+(hi[0]-lo[0])*frac+rng.uniform(-.008,.008)),float(rng.uniform(-.13,-.065)),.46] for name,frac in zip(names,[.18,.5,.82])}
            for obj in env.env.scene.movable_objects:
              obj.initial_pos=np.array(layout[obj.name]);obj.initial_orn=np.array(p.getQuaternionFromEuler([0,0,rng.uniform(-np.pi,np.pi)]))
            obs=env.reset()
            # Only standard motor holding and natural gravity/contact settle at initialization.
            for _ in range(24): obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
            root=env.evaluator_snapshot();rid=f'{envname}_{split}_{i:02d}';d=OUT/'roots'/rid;d.mkdir(parents=True)
            np.savez_compressed(d/'reset_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs'])
            save_npz(d/'initial_public.npz',obs)
            for camera,pixels in obs['rgb_obs'].items():Image.fromarray(pixels).save(d/f'{camera}.png')
            color=['red','blue','pink'][i%3]
            rows.append({'root_id':rid,'split':split,'environment':envname,'seed':seed,'chain_color':color,'reset_sha256':sha(d/'reset_evaluator_only.npz')})
            if split=='development' and i==0:
              table=env.env.scene.fixed_objects[0]; robot=env.env.robot
              geometry={'initial_robot':obs['robot_obs'].tolist(),'table_uid':table.uid,'robot_uid':robot.robot_uid,'links':[], 'objects':{o.name:{'uid':o.uid,'pose':o.get_state().tolist(),'aabb':p.getAABB(o.uid,physicsClientId=env.env.cid),'shape':p.getCollisionShapeData(o.uid,-1,physicsClientId=env.env.cid)} for o in env.env.scene.movable_objects}}
              for j in range(p.getNumJoints(table.uid,physicsClientId=env.env.cid)):
                q=p.getJointInfo(table.uid,j,physicsClientId=env.env.cid)
                geometry['links'].append({'index':j,'joint':q[1].decode(),'link':q[12].decode(),'axis':q[13],'limits':q[8:10],'world':p.getLinkState(table.uid,j,physicsClientId=env.env.cid),'aabb':p.getAABB(table.uid,j,physicsClientId=env.env.cid)})
              (d/'geometry_private.json').write_text(json.dumps(geometry,indent=2,default=lambda x:x.decode() if isinstance(x,bytes) else str(x)))
        finally:env.close()
    (OUT/'roots_manifest.json').write_text(json.dumps(rows,indent=2))
    assert len({r['reset_sha256'] for r in rows})==27
    for f in [OUT/'protocol.json',OUT/'roots_manifest.json',*OUT.glob('roots/*/reset_evaluator_only.npz')]:f.chmod(0o444)
    hand=json.loads((P/'runs/continuation_handoff.json').read_text())
    hand.update(stage='physical_oracle_v1_development',priority='User authorized independent GT-assisted physical controller. Keep formal experiment held.',next='Develop on9 frozen nativeABC roots; freeze code then18 acceptance chains and negative cases.',latest_controller_protocol=str(OUT/'protocol.json'),active_pipeline='physical_oracle_v1',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (P/'runs/continuation_handoff.json').write_text(json.dumps(hand,indent=2))
    print(json.dumps({'frozen_roots':len(rows),'protocol':str(OUT/'protocol.json')}))
if __name__=='__main__':main()
