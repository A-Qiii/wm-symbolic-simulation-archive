from pathlib import Path
import json,gzip,hashlib,sys,argparse,fcntl,time
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_causal_windows_20261014';B=P/'data/predicate_causal_replay_20261014';sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
C=['red','blue','pink']
def read(path):return json.loads(Path(path).read_text())
def save(path,obj):Path(path).write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def events(x):return [i for i in range(1,len(x))if x[i]!=x[i-1]]
def collect(root,protocol):
    folder=B/'episodes'/(root['root_id']+'__grasp');folder.mkdir(exist_ok=False);(folder/'public').mkdir();trace=[];checks=[];source=None;available={};start=time.monotonic();env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True)
    try:
        for obj in env.env.scene.movable_objects:
            val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(val[3:]))
        for door in env.env.scene.doors:
            if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
        obs=env.reset();ctl=PhysicalOracle(env);OmegaConf.save(env.config,folder/'complete_config.yaml',resolve=True);np.savez_compressed(folder/'initial_state_private.npz',**env.env.get_state_obs());save(folder/'root_provenance.json',root)
        if root['origin']=='replay':
            sd=Path(root['source_trajectory']);source=json.load(gzip.open(sd/'trace_private.json.gz','rt'))
            for row in read(sd/'samples_manifest.json'):
                for idx,pp in zip(row['history_frame_indices'],row['public_frames']):available[idx]=pp
        def record(ob,action):
            i=len(trace);lab=label(env);info=env.env.get_info();rb=info['robot_info'];objects={};raw={}
            for c in C:
                obj=ctl.objects['block_'+c];cs=[cc for cc in p.getContactPoints(bodyA=obj.uid,physicsClientId=env.env.cid)if cc[9]>.01];raw['block_'+c]=[list(z)for z in cs];rc=[z for z in cs if z[2]==ctl.robot.robot_uid];links={z[4]for z in rc};bi=all(links&family for family in ctl.finger_families);ex=[z for z in cs if z[2]!=ctl.robot.robot_uid];pos=ctl.xyz('block_'+c);pixels={cam:int((((mask&((1<<24)-1))==obj.uid)&(mask>=0)).sum())for cam,mask in ob['segmentation'].items()}
                objects[c]={'position_m':pos.tolist(),'bilateral_finger_force':bool(bi),'external_support_contacts':len(ex),'robot_force_contacts':len(rc),'held_contact_audit':bool(bi and not ex),'tcp_distance_m':float(np.linalg.norm(pos-np.array(rb['tcp_pos']))),'visible_pixels':pixels,'object_uid_evaluator_only':obj.uid}
                assert bool(lab['holding_derived'][C.index(c)])==objects[c]['held_contact_audit']
            row={'frame':i,'action':action,'labels':lab,'robot_obs':ob['robot_obs'].tolist(),'tcp_pos':list(rb['tcp_pos']),'gripper_command':int(rb['gripper_action']),'gripper_width_m':float(rb['gripper_opening_width']),'objects_audit':objects,'contacts_private':raw,'scene_obs_private':env.env.scene.get_obs().tolist(),'object_pose_state_private':{c:ctl.objects['block_'+c].get_state().tolist()for c in C}};trace.append(row);save_npz(folder/'public'/f'{i:04d}.npz',ob)
            if source is not None and i in available:
                with np.load(available[i])as z:
                    cr={'frame':i,'source':available[i],'rgb':{},'mask':{}}
                    uid=objects[root['target_color']]['object_uid_evaluator_only']
                    for cam in ['static','gripper']:
                        nr=ob['rgb_obs']['rgb_'+cam];old=z['rgb_'+cam];diff=np.abs(nr.astype(float)-old.astype(float));cr['rgb'][cam]={'bitwise_equal':bool(np.array_equal(nr,old)),'mean_abs':float(diff.mean()),'max_abs':float(diff.max())};newmask=ob['segmentation'][cam];oldmask=z['seg_'+cam];a=newmask==uid;b=oldmask==uid;union=int((a|b).sum());cr['mask'][cam]={'bitwise_equal':bool(np.array_equal(newmask,oldmask)),'target_IoU':float((a&b).sum()/union)if union else 1.}
                checks.append(cr)
            if i==24:
                state=env.env.get_state_obs();np.savez_compressed(folder/'settled_state_private.npz',**state)
                if source is not None:
                    with np.load(Path(root['source_trajectory'])/'settled_root_private.npz')as z:save(folder/'settled_comparison.json',{k:float(np.abs(state[k]-z[k]).max())for k in ['robot_obs','scene_obs']})
        record(obs,source[0]['action']if source is not None else{'phase':'reset_snapshot','source':'initial reset only'})
        if source is not None:
            with physics_guard(env):
                for row in source[1:]:
                    act=row['action'];record(env.step([np.array(act['target_xyz']),np.array(act['target_quaternion']),int(act['gripper'])]),act)
            result={'status':'replayed_original_actions','actions':len(source)-1,'source_trace_sha256':sha(Path(root['source_trajectory'])/'trace_private.json.gz')}
        else:
            ctl.recorder=record;ctl.phase='reset_settle'
            with physics_guard(env):ctl.hold(24,1);result=ctl.run('lift_'+root['target_color']+'_block_table')
        with gzip.open(folder/'trace_private.json.gz','wt')as f:json.dump(plain(trace),f)
        replay=None
        if source is not None:
            assert len(trace)==len(source);nr=np.array([r['robot_obs']for r in trace]);old=np.array([r['robot_obs']for r in source]);dv=np.abs(nr-old);pos=max(float(np.linalg.norm(np.array(n['objects_audit'][c]['position_m'])-np.array(o['objects_audit'][c]['position_m'])))for n,o in zip(trace,source)for c in C);eventchecks=[]
            for c in C:
                for k in ['held_contact_audit','bilateral_finger_force','external_support_contacts']:
                    aa=[bool(z['objects_audit'][c][k])for z in trace];bb=[bool(z['objects_audit'][c][k])for z in source];aev=events(aa);bev=events(bb);shift=max([abs(a-b)for a,b in zip(aev,bev)]or[0])if len(aev)==len(bev)and aa[0]==bb[0]else 999;eventchecks.append({'colour':c,'fact':k,'old':bev,'new':aev,'max_shift':shift,'same_values':aa==bb})
            tol=protocol['tolerances_preexecution'];settled=read(folder/'settled_comparison.json');rgbmax=max(z['mean_abs']for r in checks for z in r['rgb'].values());iou=min(z['target_IoU']for r in checks for z in r['mask'].values());numeric={'tcp_position_m':float(np.linalg.norm(nr[:,:3]-old[:,:3],axis=1).max()),'object_position_m':pos,'robot_angles_rad':float(dv[:,[3,4,5,7,8,9,10,11,12,13]].max()),'aperture_m':float(dv[:,6].max()),'all_robot_abs':float(dv.max()),'settled_max_abs':max(settled.values())};within=numeric['tcp_position_m']<=tol['max_TCP_or_object_position_m']and pos<=tol['max_TCP_or_object_position_m']and numeric['robot_angles_rad']<=tol['max_robot_or_object_angle_rad']and numeric['aperture_m']<=tol['max_gripper_width_m']and numeric['settled_max_abs']<=tol['max_settled_scene_or_robot_vector_abs']and max(z['max_shift']for z in eventchecks)<=tol['event_transition_max_shift_controls']and rgbmax<=tol['saved_RGB_mean_absolute_pixel_error_0_255']and iou>=tol['saved_target_mask_IoU_min'];exact=max(numeric.values())<=tol['exact_numeric_max_abs']and all(z['same_values']for z in eventchecks)and all(z['bitwise_equal']for r in checks for typ in ['rgb','mask']for z in r[typ].values());replay={'within_predefined_tolerance':bool(within),'exact_on_all_available_checks':bool(exact),'identity':'verified_replay_matches_available_original_records'if exact else'new_replay_trajectory_not_original_frames','numeric_errors':numeric,'events':eventchecks,'saved_frame_checks':checks,'action_sequence_identical':True,'complete_initial_state_checkpoint_frame24':settled};save(folder/'replay_verification.json',replay)
            if not within:raise RuntimeError('REPLAY_OUTSIDE_FROZEN_TOLERANCE: preserved evidence at '+str(folder))
        target=root['target_color'];j=C.index(target)
        def visible(i):return i>=3 and all(max(trace[k]['objects_audit'][target]['visible_pixels'].values())>=4 for k in range(i-3,i+1))
        def valid(i):return visible(i)and trace[i]['labels']['categorical_valid'][j]and trace[i]['labels']['categorical_valid'][3]
        hard=[i for i,t in enumerate(trace)if valid(i)and t['action']['phase']=='descend_named_block'and t['gripper_command']==-1 and t['objects_audit'][target]['bilateral_finger_force']and t['objects_audit'][target]['external_support_contacts']>0 and t['objects_audit'][target]['tcp_distance_m']<.075]
        ordinary=[i for i,t in enumerate(trace)if valid(i)and t['action']['phase']=='descend_named_block'and t['gripper_command']==1 and not t['objects_audit'][target]['robot_force_contacts']and t['objects_audit'][target]['external_support_contacts']>0 and t['objects_audit'][target]['tcp_distance_m']<.075]
        positives=[i for i,t in enumerate(trace)if valid(i)and t['action']['phase']=='verify_stable_completion'and t['objects_audit'][target]['held_contact_audit']]
        motion=[];i=0
        while i<len(trace):
            if trace[i]['action']['phase']!='lift_named_block'or not trace[i]['objects_audit'][target]['held_contact_audit']:i+=1;continue
            a=i
            while i+1<len(trace)and trace[i+1]['action']['phase']=='lift_named_block'and trace[i+1]['objects_audit'][target]['held_contact_audit']:i+=1
            ts=trace[a:i+1];tcp=np.array([t['tcp_pos']for t in ts]);obj=np.array([t['objects_audit'][target]['position_m']for t in ts]);move=float(np.linalg.norm(tcp[-1]-tcp[0]));om=float(np.linalg.norm(obj[-1]-obj[0]));drift=float(np.linalg.norm((obj-tcp)-(obj-tcp)[0],axis=1).max());motion.append({'start':a,'end':i,'TCP_motion_m':move,'object_motion_m':om,'relative_drift_m':drift,'verified':move>=.02 and om>=.02 and drift<=.015});i+=1
        verified=any(q['verified']for q in motion);records=[];gaps=[]
        for condition,candidates,pick in [('hard_negative',hard,0),('ordinary_negative',ordinary,-1),('held_positive',positives,-1)]:
            if not candidates or(condition=='held_positive'and not verified):gaps.append(condition);continue
            anchor=candidates[pick];sd=folder/'samples'/condition;sd.mkdir(parents=True);pp=[str(folder/'public'/f'{k:04d}.npz')for k in range(anchor-3,anchor+1)];lp=sd/'labels_private.json';save(lp,{'frames':trace[anchor-3:anchor+1],'anchor_frame':anchor,'current_frame_label_semantics':'causal_anchor_v2','physical_held_definition_unchanged':True,'motion_audit_private':motion});records.append({'sample_id':root['root_id']+'__'+condition,'root_id':root['root_id'],'origin_root_id':root['root_id'],'trajectory_id':root['root_id']+'__grasp','environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'target_color':target,'condition':condition,'anchor_frame':anchor,'history_frame_indices':list(range(anchor-3,anchor+1)),'public_frames':pp,'labels_private':str(lp),'export_version':'causal_anchor_v2','origin':root['origin'],'replay_identity':replay['identity']if replay else'fresh_development','replay_is_new_independent_root':False if replay else True})
        summary={'root_id':root['root_id'],'origin':root['origin'],'split':root['split'],'dev_role':root['dev_role'],'frames':len(trace),'controller_result':result,'replay_verification':replay,'motion_verified':verified,'motion_audit':motion,'candidate_anchor_counts':{'hard':len(hard),'ordinary':len(ordinary),'held':len(positives)},'exported_windows':len(records),'complete_matched_triplet':not gaps,'gaps':gaps,'physics_guard_passed':True,'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'public_input_files':len(trace),'seconds':time.monotonic()-start};save(folder/'samples_manifest.json',records);save(folder/'summary.json',summary);return records,summary
    finally:env.close()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','train','dev'],required=True);args=ap.parse_args();protocol=read(E/'configs/protocol.json')
    for f,h in protocol['frozen_sha256'].items():assert sha(f)==h
    if args.phase!='pilot':assert read(E/'results/pilot_gate.json')['passed']
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);roots=read(B/'roots_manifest.json');manifest=read(B/'data_manifest.json');done={r['root_id']for r in manifest['trajectories']}
    for r in roots:
        if r['root_id']in done:continue
        if args.phase=='pilot'and r['root_id']not in protocol['pilot_roots']:continue
        if args.phase=='train'and r['split']!='train':continue
        if args.phase=='dev'and r['split']!='dev':continue
        rows,summary=collect(r,protocol);manifest['records']+=rows;manifest['trajectories'].append(summary);save(B/'data_manifest.json',manifest);save(E/'results/progress.json',{'stage':args.phase+'_collection','completed_roots':len(manifest['trajectories']),'last_root':r['root_id'],'training_started':False});print(json.dumps({'root':r['root_id'],'frames':summary['frames'],'complete':summary['complete_matched_triplet'],'replay':summary['replay_verification']['exact_on_all_available_checks']if summary['replay_verification']else None,'gaps':summary['gaps']},ensure_ascii=False),flush=True)
    if args.phase=='pilot':
        ss=[s for s in manifest['trajectories']if s['root_id']in protocol['pilot_roots']];passed=len(ss)==2 and all(s['replay_verification']['within_predefined_tolerance']and s['complete_matched_triplet']and s['motion_verified']for s in ss);save(E/'results/pilot_gate.json',{'passed':passed,'root_ids':[s['root_id']for s in ss],'checks':'frozen state/event/image tolerances, full causal public input, actual hard andheld triplet,motor-only guard','no_tolerance_change':True});assert passed
    print('COLLECTION_PHASE_COMPLETE',args.phase)
if __name__=='__main__':main()
