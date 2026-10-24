"""Independent frozen-controller acceptance; no parameter selection here."""
from pathlib import Path
import argparse, json, hashlib, datetime
from contextlib import contextmanager
import numpy as np
import pybullet as p
import hydra, imageio.v2 as imageio
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter, save_npz
from wm_symbolic.physical_oracle import PhysicalOracle

P=Path('/home/robot/research/projects/wm_symbolic');BASE=P/'runs/physical_oracle_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@contextmanager
def physics_guard(env):
    """Runtime guard enforces physical motor-only skill execution."""
    forbidden=['resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','changeDynamics','setCollisionFilterPair','setCollisionFilterGroupMask','createConstraint','removeConstraint','changeConstraint','applyExternalForce','applyExternalTorque','setGravity','setPhysicsEngineParameter','resetSimulation']
    original={name:getattr(p,name) for name in forbidden+['setJointMotorControl2']}
    def denied(*args,**kwargs):raise RuntimeError('Nonphysical simulator mutation forbidden during skill')
    def motor(*args,**kwargs):
        body=kwargs.get('bodyIndex',args[0] if args else None)
        if body!=env.env.robot.robot_uid:raise RuntimeError('Only robot motors may be actuated')
        return original['setJointMotorControl2'](*args,**kwargs)
    try:
        for name in forbidden:setattr(p,name,denied)
        p.setJointMotorControl2=motor
        yield
    finally:
        for name,value in original.items():setattr(p,name,value)

def one_episode(row,case,out,task_oracle):
    rid=row['root_id'];env=CalvinAdapter(P,row['environment'],row['seed'],tabletop=True)
    d=out/rid/case;d.mkdir(parents=True,exist_ok=False);results=[]
    try:
        rootpath=BASE/'roots'/rid/'reset_evaluator_only.npz'
        assert sha(rootpath)==row['reset_sha256']
        env.reset({'environment':row['environment'],**dict(np.load(rootpath))})
        save_npz(d/'actual_initial_public.npz',env.observe())
        named='block_'+row['chain_color']; other='blue' if row['chain_color']!='blue' else 'pink'
        if case=='chain':requests=[('open_drawer','success'),('lift_'+row['chain_color']+'_block_table','success'),('place_in_drawer','success'),('close_drawer','success')]
        else:requests=[('place_in_drawer','requires_exactly_one_held_block'),('close_drawer','drawer_already_closed'),('lift_'+row['chain_color']+'_block_table','success'),('place_in_drawer','requires_open_drawer'),('open_drawer','requires_empty_gripper'),('lift_'+other+'_block_table','requires_empty_gripper')]
        for i,(skill,expected) in enumerate(requests):
            sdir=d/f'{i:02d}_{skill}';sdir.mkdir();initial=env.observe();startinfo=env.env.get_info();startscene=env.env.scene.get_obs().copy()
            startobjects={o.name:o.get_state().copy() for o in env.env.scene.movable_objects}
            writers={k:imageio.get_writer(str(sdir/(k+'.mp4')),fps=30,macro_block_size=1,codec='libx264',quality=7) for k in initial['rgb_obs']}
            for k,w in writers.items():w.append_data(initial['rgb_obs'][k])
            save_npz(sdir/'initial_public.npz',initial);control=[];trace=[]
            robotuid=env.env.robot.robot_uid; tableuid=env.env.scene.fixed_objects[0].uid
            target=named if skill.startswith('lift_') or skill=='place_in_drawer' else None
            if skill.startswith('lift_'):target='block_'+skill.split('_')[1]
            maxdrift={k:0. for k in startobjects};peakpenetration=0.;peakforce=0.;unrequestedcontacts=0;contactframes=0
            def record(obs,act):
                nonlocal peakpenetration,peakforce,unrequestedcontacts,contactframes
                control.append(act)
                for k,w in writers.items():w.append_data(obs['rgb_obs'][k])
                objects={o.name:o.get_state() for o in env.env.scene.movable_objects}
                contacts=[c for c in p.getContactPoints(physicsClientId=env.env.cid) if c[9]>.01]
                robotcontacts=[c for c in contacts if c[1]==robotuid or c[2]==robotuid]
                if robotcontacts:contactframes+=1
                for c in robotcontacts:
                    peakpenetration=max(peakpenetration,-c[8]);peakforce=max(peakforce,c[9])
                    otherbody=c[2] if c[1]==robotuid else c[1];otherlink=c[4] if c[1]==robotuid else c[3]
                    allowed=otherbody==tableuid and otherlink==controller.links['drawer_link'] and skill in ['open_drawer','close_drawer']
                    allowed|=target is not None and otherbody==controller.objects[target].uid
                    if not allowed:unrequestedcontacts+=1
                for name,value in objects.items():maxdrift[name]=max(maxdrift[name],float(np.linalg.norm(value[:3]-startobjects[name][:3])))
                trace.append({'robot_obs':obs['robot_obs'].tolist(),'objects':{k:v.tolist() for k,v in objects.items()},'contacts':[list(c) for c in contacts], 'scene_obs':env.env.scene.get_obs().tolist()})
            controller=PhysicalOracle(env,record)
            try:
                with physics_guard(env):r=controller.run(skill)
            finally:
                for w in writers.values():w.close()
            official=skill in task_oracle.get_task_info_for_set(startinfo,env.env.get_info(),{skill})
            # Drawer transport of its contained object is an intended effect of
            # closing, unlike moving unrelated tabletop objects.
            allowedmove={target} if target else ({named} if case=='chain' and skill=='close_drawer' else set())
            max_unrequested=max((v for k,v in maxdrift.items() if k not in allowedmove),default=0.)
            finalscene=env.env.scene.get_obs().copy()
            r.update(skill=skill,expected=expected,root=rid,environment=row['environment'],case=case,official_success=official,
                     correct_refusal=r['status']=='refused' and r['reason']==expected and r['controls']==0 and np.array_equal(startscene,finalscene),
                     physics_guard_passed=True,max_object_displacements_m=maxdrift,max_unrequested_object_displacement_m=max_unrequested,
                     robot_contact_frames=contactframes,unrequested_robot_contact_samples=unrequestedcontacts,
                     max_robot_contact_penetration_m=peakpenetration,max_robot_contact_force_N=peakforce,
                     unrequested_device_state_change=bool(np.max(np.abs(finalscene[[0,2,3,4,5]]-startscene[[0,2,3,4,5]]))>.003),
                     public_observation_keys=sorted(initial),controller=controller.VERSION)
            r['conservative_pass']=bool((r['correct_refusal'] if expected!='success' else r['status']=='success' and official) and max_unrequested<.01 and peakpenetration<.01 and not r['unrequested_device_state_change'])
            (sdir/'result.json').write_text(json.dumps(r,indent=2));(sdir/'controls.json').write_text(json.dumps(control));(sdir/'trace_private.json').write_text(json.dumps(trace));save_npz(sdir/'final_public.npz',env.observe())
            results.append(r);(d/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:r[k] for k in ['root','case','skill','status','reason','controls','official_success','correct_refusal','conservative_pass','max_unrequested_object_displacement_m','max_robot_contact_penetration_m']}),flush=True)
            if expected=='success' and r['status']!='success':break
        final=PhysicalOracle(env)
        outcome={'root':rid,'case':case,'requested':len(requests),'attempted':len(results),'complete':len(results)==len(requests) and all(x['conservative_pass'] for x in results),
                 'named_block_in_drawer_at_chain_end':final.in_drawer(named) if case=='chain' else None,'drawer_position':final.joint('drawer_link')}
        if case=='chain':outcome['complete']&=bool(outcome['named_block_in_drawer_at_chain_end'] and outcome['drawer_position']<.02)
        (d/'episode_summary.json').write_text(json.dumps(outcome,indent=2));return outcome
    finally:env.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development','acceptance'],required=True);parser.add_argument('--name',required=True);args=parser.parse_args()
    out=BASE/args.name;out.mkdir(exist_ok=False)
    freeze=json.loads((BASE/'controller_freeze.json').read_text())
    assert sha(P/'src/wm_symbolic/physical_oracle.py')==freeze['controller_sha256']
    assert sha(Path(__file__))==freeze['evaluation_script_sha256']
    rows=[r for r in json.loads((BASE/'roots_manifest.json').read_text()) if r['split']==args.split]
    oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
    results=[]
    for row in rows:
      for case in ['chain','negative_preconditions']:
        results.append(one_episode(row,case,out,oracle));(out/'episode_summaries.json').write_text(json.dumps(results,indent=2))
    assert sha(P/'src/wm_symbolic/physical_oracle.py')==freeze['controller_sha256']
    (out/'COMPLETE.json').write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'episodes':len(results),'code_unchanged':True},indent=2))
if __name__=='__main__':main()
