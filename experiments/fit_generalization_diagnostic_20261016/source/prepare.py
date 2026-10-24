from config import *
import sys,xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
sys.path.insert(0,str(REP/'source'))
from labels_v2_1 import evaluate,SPEC

def main():
    assert not(E/'configs/protocol.json').exists()
    for sub in ['configs','source','results','logs','data/public','data/_private']:(E/sub).mkdir(parents=True,exist_ok=True)
    R.mkdir(parents=True,exist_ok=True);sealed=read(STUDY/'configs/delivery_manifest.json')
    for path,h in sealed['sha256'].items():assert sha(path)==h,path
    sources={};geometries={};configs={}
    for env in ['A','B','C']:
        path=P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml';cfg=OmegaConf.load(path);configs[env]=cfg;sources[str(path)]=sha(path);urdf=P/'vendor/calvin/calvin_env/data'/cfg.objects.fixed_objects.table.file;sources[str(urdf)]=sha(urdf);tree=ET.parse(urdf).getroot();joint=tree.find("joint[@name='base__drawer']");assert joint.find('parent').attrib['link']=='base_link';origin=np.fromstring(joint.find('origin').attrib['xyz'],sep=' ')*cfg.global_scaling;rot=Rotation.from_euler('xyz',np.fromstring(joint.find('origin').attrib['rpy'],sep=' '));axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ');base=Rotation.from_euler('xyz',cfg.objects.fixed_objects.table.initial_orn);basepos=np.array(cfg.objects.fixed_objects.table.initial_pos);link=[j.find('child').attrib['link']for j in tree.findall('joint')].index('drawer_link');assert link==3;geometries[env]=(basepos,base,origin,rot,axis,link)
    def drawer(env,q):
        basepos,base,origin,rot,axis,link=geometries[env];return basepos+base.apply(origin+rot.apply(axis*q)),(base*rot).as_quat(),link
    # Validate offline URDF kinematics against already recorded full snapshots, no physics instantiated.
    maxpos=0.;maxangle=0.;validated=0
    episodes=read(PRIOR/'results/collection_ABC.json')
    for ep in episodes:
        env=ep['root_id'][0];path=Path(ep['folder'])/'_private/states.json';sources[str(path)]=sha(path)
        for row in read(path):
            snap=row['physical'];pos,quat,link=drawer(env,snap['drawer_q']);maxpos=max(maxpos,float(np.abs(pos-snap['drawer_world_position']).max()));maxangle=max(maxangle,float((Rotation.from_quat(quat).inv()*Rotation.from_quat(snap['drawer_world_quaternion'])).magnitude()));assert link==snap['drawer_link']and snap['table_uid']==5
            converted=snap|{'drawer_world_position':pos.tolist(),'drawer_world_quaternion':quat.tolist()};a=evaluate(converted);assert a['y']==row['labels']['y']and a['v']==row['labels']['v'];validated+=1
    # Recorded Bullet link-frame cache and current joint-q differ by up to9um on moving frames.
    # All1100 original classifications above must still match; keep a10um uncertainty guard below.
    assert maxpos<1e-5 and maxangle<1e-6
    oldstates=read(REP/'data/public/states.json');oldpairs=read(REP/'data/public/pairs.json');oldout=read(REP/'data/_private/outcomes.json');newstates=read(PRIOR/'data/ABC/public/states.json');newpairs=read(PRIOR/'data/ABC/public/pairs.json');newlabels=read(PRIOR/'data/ABC/_private/labels.json');newout=read(PRIOR/'data/ABC/_private/outcomes.json');oldarr=dict(np.load(PACK/'inputs.npz'));newarr=dict(np.load(PRIOR/'data/ABC/public/inputs.npz'));matched=read(STUDY/'configs/chain_matching.json');states=[];lookup={};labels={};arrays={k:[]for k in ['objects','scene','z']};trajectories=[];truth=[];legacy_labels=0
    def add(group,i):
        nonlocal legacy_labels
        key=(group,i)
        if key in lookup:return lookup[key]
        idx=len(states);lookup[key]=idx
        if group=='old':
            st=oldstates[i];src=Path(st['source']);frame=int(src.stem);root=st['root_id'];private=src.parent.parent/'_private'/f'{root}__t{frame:05d}.json';sources[str(private)]=sha(private);raw=read(private)['frames'][-1];assert raw['frame']==frame
            with np.load(st['path'])as obs:assert np.array_equal(obs['robot_obs'],np.asarray(raw['robot_obs'],dtype=obs['robot_obs'].dtype))
            with np.load(src)as obs:assert np.array_equal(obs['robot_obs'],np.asarray(raw['robot_obs']))
            scene=np.array(raw['scene_obs_private']);assert len(scene)==24;env=root[0];order=list(configs[env].objects.movable_objects.keys());assert set(order)=={'block_pink','block_blue','block_red'};objects={}
            for j,name in enumerate(order):
                color=name.split('_')[1];pose=scene[6+6*j:12+6*j];assert np.allclose(pose[:3],raw['objects_audit'][color]['position_m'],atol=1e-10);objects[name]={'uid':raw['objects_audit'][color]['object_uid_evaluator_only'],'position':pose[:3].tolist(),'quaternion':Rotation.from_euler('xyz',pose[3:]).as_quat().tolist()}
            contacts=[x for cc in raw['contacts_private'].values()for x in cc];seen=set();contacts=[x for x in contacts if not(json.dumps(x)in seen or seen.add(json.dumps(x)))];q=raw['drawer_position_m'];assert abs(scene[1]-q)<1e-10;pos,quat,link=drawer(env,q)
            assert all(x[9]>.01 for x in contacts)
            snap={'objects':objects,'contacts':contacts,'held':['block_'+c for c in COLORS if raw['objects_audit'][c]['held']],'table_uid':5,'drawer_link':link,'drawer_world_position':pos.tolist(),'drawer_world_quaternion':quat.tolist(),'drawer_q':q};lab=evaluate(snap);assert lab['y'][:3]==raw['labels']['holding_derived'];legacy_labels+=1
            # Legacy collector saved only force>.01N contacts. Missing weak contacts cannot establish absence.
            for j,color in enumerate(COLORS):
                obj=lab['objects'][color]
                if not obj['outside']and not obj['held']and not obj['certain_contained_load_bearing_paths']:lab['v'][3+j]=False
                if not obj['held']and min(abs(obj['minimum_side_rim_clearance_m']-.002),abs(obj['minimum_side_rim_clearance_m']+.002),abs(obj['floor_clearance_m']+.003))<=1e-5:lab['v'][3+j]=False
            lab['provenance']='Offline same-definition reconstruction from recorded full XYZ/Euler,contacts,held,drawer q and native URDF; no physics execution. Legacy contacts filtered at>.01N: no-certain-path,not-held,not-outside In remains unknown.';lab['source']=str(private)
            arr=oldarr
        else:st=newstates[i];lab=newlabels[str(i)];arr=newarr
        states.append({'index':idx,'root_id':st['root_id'],'source_group':group,'source_index':i,'path':st.get('path',st.get('public_path'))});labels[str(idx)]=lab
        for k in arrays:arrays[k].append(arr[k][i])
        return idx
    for rp in matched['root_pairs']:
        root=rp['old_root'];rs=[oldpairs[i]for i in rp['old_pairs']];steps=[];private=[]
        for slot,r in enumerate(rs):
            assert r['split']=='train';steps.append({'slot':slot,'skill':r['skill'],'action':r['action'],'before':add('old',r['before']),'after':add('old',r['after'])});private.append(oldout[r['id']])
        trajectories.append({'group':'old_AB','root_id':root,'color':rp['target'],'branch':'normal','steps':steps});truth.append({'root_id':root,'branch':'normal','outcomes':private})
    groups={}
    for row in newpairs:groups.setdefault((row['root_id'],row['branch']),[]).append(row)
    for (root,branch),rs in groups.items():
        rs.sort(key=lambda r:r['slot']);assert len(rs)==4;split=rs[0]['split'];group={'train':'new_AB','dev_select':'C_select','dev_validate':'C_validate'}[split];steps=[];private=[]
        for row in rs:steps.append({k:row[k]for k in ['slot','skill','action']}|{'before':add('new',row['before']),'after':add('new',row['after'])});private.append(newout[row['id']])
        trajectories.append({'group':group,'root_id':root,'color':rs[0]['target'],'branch':branch,'steps':steps});truth.append({'root_id':root,'branch':branch,'outcomes':private})
    for t in trajectories:
        assert len(t['steps'])==4
        for a,b in zip(t['steps'],t['steps'][1:]):assert a['after']==b['before']
        for r in t['steps']:
            action=np.zeros(8);family='lift'if r['skill'].startswith('lift_')else r['skill'].split('_')[0];action[['open','lift','place','close'].index(family)]=1
            if family=='lift':action[4+COLORS.index(t['color'])]=1
            action[7]=1;assert r['action']==action.tolist()
    assert len(trajectories)==60
    np.savez_compressed(E/'data/public/inputs.npz',**{k:np.stack(v)for k,v in arrays.items()});save(E/'data/public/states.json',states);save(E/'data/public/trajectories.json',trajectories);save(E/'data/_private/labels.json',labels);save(E/'data/_private/outcomes.json',truth)
    save(E/'results/source_alignment.json',{'prior_sealed_files':len(sealed['sha256']),'offline_drawer_validation_states':validated,'max_drawer_position_error_m':maxpos,'max_drawer_rotation_error_rad':maxangle,'offline_legacy_labels':legacy_labels,'legacy_full9':sum(all(labels[str(i)]['v'])for i,s in enumerate(states)if s['source_group']=='old'),'all_states':len(states),'legacy_public_robot_obs_matches_last_private_frame':True,'action_matches_skill_color_and96posthold':True,'all_adjacency_verified':True,'no_physics_instantiated':True,'label_definition':SPEC})
    protocol={'frozen_utc':now(),'scope':'Frozen model inference only. No optimization, new simulation, data collection, threshold or checkpoint reselection.','models':read(STUDY/'configs/choices_frozen.json'),'normal_roots':{'old_AB':12,'new_AB':12,'C_select':6,'C_validate':6},'cross_evaluation':'All4selected models on both training source cohorts and C cohorts. Own-source AB identifies fit; cross-source AB is descriptive and differs in root distribution.','paths':['actual successor readout','true predecessor one-step','full4 from root initial only','exact3-skill training window start0','exact3-skill training window start1'],'refusals':'newAB/C held_closed branches separate; not pooled with normal','C_role':'Cselect was used for model selection; Cvalidate was held out from training/choice but historically seen. Neither is a new test.','loss_diagnostic':'Read sparse logged loss; frozen inference of all20saved checkpoints on own training windows and exact frequency-weighted single-step schedule. Also show C3/4-step recursive curves diagnostically as newly authorized; never change the previously selected checkpoints or claim newly chosen best C as formal performance.','truth':'same9propositions; old60 endpoint states reconstructed offline from existing XYZ/Euler+contacts and native assets, native transform validated against1100 prior full snapshots. Contact censoring and10um transformation uncertainty are conservatively masked, never filled false.','representation':'current2RGB+current15D robot_obs only; robot_history field name is misleading. No observation history and no recurrent posterior. Transition recursion carries its predicted latent only.','length':'Trained windows3skills with losses at local steps2/3. The second window supervises globalstep4 from true post-open state at localdepth3; no loss at depth4 from initial.'}
    save(E/'configs/protocol.json',protocol)
    paths=list((E/'source').glob('*.py'))+[E/'configs/protocol.json',E/'data/public/inputs.npz',E/'data/public/states.json',E/'data/public/trajectories.json',E/'data/_private/labels.json',E/'data/_private/outcomes.json',STUDY/'configs/choices_frozen.json',STUDY/'configs/chain_matching.json',REP/'source/model.py',REP/'source/runtime.py',REP/'source/features.py',REP/'source/representation.py',REP/'source/labels_v2_1.py',PACK/'inputs.npz',REP/'data/public/states.json',REP/'data/public/pairs.json',PRIOR/'data/ABC/public/inputs.npz',PRIOR/'data/ABC/public/states.json',PRIOR/'data/ABC/public/pairs.json',PRIOR/'data/ABC/_private/labels.json',P/'experiments/state_predicates_long_pairs_20261014/source/collect.py']+[Path(v['path'])for v in protocol['models']['selected'].values()]+[Path(v['path'])for v in protocol['models']['query_heads'].values()]
    for seed in [0,1]:
        paths.append(STUDY/'configs'/f'schedule_seed{seed}.npz')
        for arm in ['old_chains','new_chains']:paths+=list((STUDY/'models'/f'{arm}_seed{seed}').glob('update*.pt'))+[STUDY/'results'/f'train_{arm}_seed{seed}.json']
    save(E/'configs/frozen.json',{'frozen_utc':now(),'sha256':sources|{str(p):sha(p)for p in paths}});print('PREPARED',len(states),len(trajectories),'legacy_labels',legacy_labels,flush=True)
if __name__=='__main__':main()
