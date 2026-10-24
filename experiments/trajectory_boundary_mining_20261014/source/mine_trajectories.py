"""A/B-only physical boundary audit. No train/dev/test prediction or simulation."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,hashlib,datetime,sys
import numpy as np
import imageio.v2 as imageio
import pybullet as pb
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/trajectory_boundary_mining_20261014';COLORS=['red','blue','pink'];HASH={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,gz=False):
    p=Path(p);b=p.read_bytes();HASH[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(gzip.decompress(b)if gz else b)
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def families():
    path=P/'vendor/calvin/calvin_env/data/franka_panda/panda_longer_finger.urdf';HASH[str(path)]=sha(path);cid=pb.connect(pb.DIRECT);u=pb.loadURDF(str(path),useFixedBase=True,physicsClientId=cid);js={i:pb.getJointInfo(u,i,physicsClientId=cid)for i in range(pb.getNumJoints(u,physicsClientId=cid))};out=[]
    for finger in [9,11]:
        group={finger}
        for _ in js:group.update(i for i,q in js.items()if q[16]in group)
        out.append(group)
    pb.disconnect(cid);return out
def state(obj,contacts,uid,robot,rob,fam):
    cs=[c for c in contacts if c[9]>.01 and uid in [c[1],c[2]]];links=set();external=0
    for c in cs:
        other,link=(c[2],c[4])if c[1]==uid else(c[1],c[3])
        if other==robot:links.add(link)
        else:external+=1
    bilateral=all(links&g for g in fam);return {'bilateral':bool(bilateral),'external':external,'held':bool(bilateral and not external),'any_robot_contact':bool(links),'tcp_distance_m':float(np.linalg.norm(np.array(obj[:3])-np.array(rob[:3]))),'gripper_width_m':float(rob[6]),'command':float(rob[-1])}
def colour_pixels(rgb,c):
    x=rgb.astype(np.float32)/255;r,g,b=x[:,:,0],x[:,:,1],x[:,:,2]
    mask=(r>.35)&(r>g*1.7)&(r>b*1.7)if c=='red'else((b>.35)&(b>r*1.7)&(b>g*1.7)if c=='blue'else(r>.35)&(b>.35)&(g<np.minimum(r,b)*.6))
    return int(mask.sum())
def runs(mask):
    out=[];i=0
    while i<len(mask):
        if not mask[i]:i+=1;continue
        a=i
        while i+1<len(mask)and mask[i+1]:i+=1
        out.append((a,i));i+=1
    return out
def main():
    for folder in ['source','configs','results','figures','logs']:(E/folder).mkdir(parents=True,exist_ok=True)
    assert not(E/'results/mining.json').exists(),'Do not overwrite the completed audit'
    fam=families();protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing A/B source trajectories only. C and D trajectory content remains unopened by this mining run.','held_semantics':'At the anchor frame: bilateral force-bearing descendant fingers, force>0.01, no external force-bearing contact. Previous frames may have different labels. No future frame is an input.','hard_negative_definition':'Anchor bilateral force-bearing fingers AND external force-bearing contact, closed command, TCP<75mm; exactly not held under unchanged physical rule. Frame history at t-3..t must exist.','original_supplement_selection':'Original predicate required all4frames to be visible, nonheld, externally supported, near, closed-command, descend_named_block. This audit separately counts transient anchor facts and stable4 requirements.','visibility':'Supplement has saved exact mask pixel counts in private audit. Core full video has only RGB colour-pixel proxy; not exact segmentation. Visibility is reported separately, never invented.','public_input_contract':'Existing reader requires original RGB, per-frame visible masks and robot_obs. Decoded MP4 is lossy and no substitute mask may be generated from private states.','training':'No training unless complete original public windows and held-out development design are feasible; no model or controller change.','finger_families':[sorted(x)for x in fam],'training_roots_only':True,'prior_preflight':'Supplement A/B preliminary trace inspection already showed3stable roots versus12anchor roots; protocol freezes the full audit definitions, not a claimed blind discovery.'}
    save(E/'configs/audit_protocol.json',protocol);(E/'configs/audit_protocol.json').chmod(0o444)
    rootmap={};trajectories=[];windows=[];trace_totals=Counter();full_motion={};physical_mismatches=[]
    core=P/'data/core1200_v1';roots=read(core/'roots_manifest.json');rootmap.update({r['root_id']:r|{'source':'core1200'}for r in roots if r['environment']in ['A','B']});records=[r for r in read(core/'data_manifest.json')['records']if r['environment']in ['A','B']]
    sup=P/'data/predicate_gap_supplement_20261014';roots=read(sup/'roots_manifest.json');sroots=[r for r in roots if r['environment']in ['A','B']];rootmap.update({r['root_id']:r|{'source':'supplement'}for r in sroots})
    tasks=[('core',r,Path(r['obs_before']).parent)for r in records]+[('supplement',r,sup/'episodes'/(r['root_id']+'__'+branch))for r in sroots for branch in r['trajectories']]
    for ti,(source,r,folder)in enumerate(tasks):
        if source=='core':
            tr=read(folder/'trace_private.json.gz',True);actions=read(folder/'controls.json');assert len(tr)==len(actions);info=read(folder/'endpoint_info_private.json');initial=info['before'];N=len(tr)+1;video_offset=1;labels=[read(folder/'labels_ownership_v2_private.json')[side]for side in ['before','after']];states={c:[]for c in COLORS};phases=['initial']+[a['phase']for a in actions]
            for c in COLORS:
                uid=initial['scene_info']['movable_objects']['block_'+c]['uid'];robot=initial['robot_info']['uid'];b=initial['robot_info'];rob=list(b['tcp_pos'])+list(b['tcp_orn'])+[b['gripper_opening_width']]+list(b['arm_joint_states'])+[b['gripper_action']];obj=initial['scene_info']['movable_objects']['block_'+c];states[c].append(state(obj['current_pos'],obj['contacts'],uid,robot,rob,fam))
                for t in tr:states[c].append(state(t['objects']['block_'+c],t['contacts'],uid,robot,t['robot_obs'],fam))
                for q,lab in [(states[c][0],labels[0]),(states[c][-1],labels[1])]:
                    if q['held']!=bool(lab['holding_derived'][COLORS.index(c)]):physical_mismatches.append({'trajectory':str(folder),'colour':c})
            available={0:[str(folder/'initial_public.npz')]};n=len(tr)
            for k in range(4):
                f=n-3+k
                if f>=0:available.setdefault(f,[]).append(str(folder/f'boundary_history_{k}.npz'))
            available.setdefault(n,[]).append(str(folder/'final_public.npz'))
            tid=r['episode_id']+'/'+folder.name;target=None
        else:
            tr=read(folder/'trace_private.json.gz',True);N=len(tr);video_offset=0;states={c:[]for c in COLORS};phases=[t['action']['phase']for t in tr];available={};target=r['target_color'];tid=folder.name
            for i,t in enumerate(tr):
                for c in COLORS:
                    o=t['objects_audit'][c];v={'bilateral':o['bilateral_finger_force'],'external':o['external_support_contacts'],'held':o['held_contact_audit'],'any_robot_contact':any(cc[9]>.01 and cc[2]==0 for cc in t['contacts_private']['block_'+c]),'tcp_distance_m':o['tcp_distance_m'],'gripper_width_m':t['gripper_width_m'],'command':t['gripper_command'],'exact_visible_pixels':o['visible_pixels']};states[c].append(v)
                    if v['held']!=bool(t['labels']['holding_derived'][COLORS.index(c)]):physical_mismatches.append({'trajectory':tid,'frame':i,'colour':c})
            sm=read(folder/'samples_manifest.json')
            for sample in sm:
                for idx,path in zip(sample['history_frame_indices'],sample['public_frames']):available.setdefault(idx,[]).append(path)
            full_motion[tid]={'root_id':r['root_id'],'target':target,'frames':[{'frame':i,'phase':phases[i],**states[target][i]}for i in range(N)]}
        trace_totals[source]+=N;found=[];run_records=[];ordinary=[]
        for c in COLORS:
            seq=states[c];mask=[q['bilateral']and q['external']>0 and q['command']==-1 and q['tcp_distance_m']<.075 for q in seq];spans=runs(mask);spanmap={i:(a,b)for a,b in spans for i in range(a,b+1)}
            for a,b in spans:run_records.append({'colour':c,'first':a,'last':b,'frames':b-a+1})
            for i,q in enumerate(seq):
                if i<3:continue
                if not q['held']and q['external']>0 and not q['any_robot_contact']and q['command']==-1 and q['tcp_distance_m']<.075:ordinary.append({'frame':i,'colour':c,'public_complete':all(j in available for j in range(i-3,i+1)),'width_m':q['gripper_width_m'],'distance_m':q['tcp_distance_m']})
                if not mask[i]:continue
                a,b=spanmap[i];orig4=None
                if source=='supplement':
                    def old(j):
                        z=seq[j];return not z['held']and z['external']>0 and max(z['exact_visible_pixels'].values())>=4 and z['tcp_distance_m']<.075 and phases[j]=='descend_named_block'and z['command']==-1
                    orig4=all(old(j)for j in range(i-3,i+1))
                exact=[seq[j].get('exact_visible_pixels')for j in range(i-3,i+1)];complete=all(j in available for j in range(i-3,i+1));w={'source':source,'root_id':r['root_id'],'environment':r['environment'],'trajectory':tid,'folder':str(folder),'colour':c,'frame':i,'phase':phases[i],'history_indices':list(range(i-3,i+1)),'history_held':[seq[j]['held']for j in range(i-3,i+1)],'history_hard':[bool(mask[j])for j in range(i-3,i+1)],'hard_run_frames':b-a+1,'stable4_physical':all(mask[j]for j in range(i-3,i+1)),'original_supplement_stable4_predicate':orig4,'robot_width_m':q['gripper_width_m'],'tcp_distance_m':q['tcp_distance_m'],'exact_visible_pixels':exact,'complete_original_public_window':complete,'public_paths':[available[j][0]for j in range(i-3,i+1)]if complete else [],'missing_public_frame_indices':[j for j in range(i-3,i+1)if j not in available]}
                if exact[-1]is not None:w['exact_anchor_visible']=max(exact[-1].values())>=4;w['exact_all4_visible']=all(max(z.values())>=4 for z in exact)
                windows.append(w);found.append(w)
        traj={'source':source,'trajectory':tid,'root_id':r['root_id'],'environment':r['environment'],'frames':N,'candidate_windows':len(found),'candidate_colours':sorted(set(w['colour']for w in found)),'hard_runs':run_records,'ordinary_close_near_windows':len(ordinary),'ordinary_examples':ordinary[:2]+ordinary[-2:]if ordinary else [],'available_original_public_frame_count':len(available),'public_video_count':2,'per_frame_original_masks_saved':False}
        if found:
            need=set(j for w in found for j in w['history_indices']);pixels={};decodes={}
            for cam in ['static','gripper']:
                vp=folder/f'rgb_{cam}.mp4';HASH[str(vp)]=sha(vp);reader=imageio.get_reader(str(vp));n=0
                try:
                    for fi,img in enumerate(reader):
                        n+=1
                        if fi in need:pixels[(fi,cam)]={c:colour_pixels(img,c)for c in COLORS}
                        if source=='supplement'and target=='red'and r['drawer_setting']=='closed'and fi in set(j for w in found for j in w['history_indices']):
                            out=E/'figures'/(tid+f'__{fi:04d}_{cam}.png');imageio.imwrite(out,img)
                finally:reader.close()
                assert n==N,(str(vp),n,N);decodes[cam]={'decoded_frames':n,'expected':N,'codec':'H264 lossy, source collector quality7 or8'}
            for w in found:
                w['rgb_colour_proxy_pixels']=[{cam:pixels[(j,cam)][w['colour']]for cam in ['static','gripper']}for j in w['history_indices']];w['rgb_proxy_anchor_visible']=max(w['rgb_colour_proxy_pixels'][-1].values())>=4;w['rgb_proxy_all4_visible']=all(max(z.values())>=4 for z in w['rgb_colour_proxy_pixels'])
            traj['video_alignment']=decodes
        trajectories.append(traj)
        if(ti+1)%60==0:print('AUDITED',ti+1,'of',len(tasks),'candidate_windows',len(windows),flush=True)
    assert not physical_mismatches
    rootrows=[]
    for rid,root in sorted(rootmap.items()):
        ww=[w for w in windows if w['root_id']==rid];rootrows.append({'root_id':rid,'environment':root['environment'],'source':root['source'],'initial_state':root.get('initial_state',root.get('drawer_setting')),'layout':root['layout'],'candidate_windows':len(ww),'candidate_colours':sorted(set(w['colour']for w in ww)),'short_run_colours':sorted(set(w['colour']for w in ww if w['hard_run_frames']<4)),'stable4_colours':sorted(set(w['colour']for w in ww if w['stable4_physical'])),'reusable_public_colours':sorted(set(w['colour']for w in ww if w['complete_original_public_window'])),'rgb_proxy_visible_colours':sorted(set(w['colour']for w in ww if w['rgb_proxy_anchor_visible']))})
    summary={}
    for source in ['core','supplement']:
        ww=[w for w in windows if w['source']==source];tt=[t for t in trajectories if t['source']==source];summary[source]={'trajectories':len(tt),'frames':sum(t['frames']for t in tt),'candidate_windows':len(ww),'candidate_roots':len(set(w['root_id']for w in ww)),'root_colour_pairs':len(set((w['root_id'],w['colour'])for w in ww)),'stable4_windows':sum(w['stable4_physical']for w in ww),'stable4_roots':len(set(w['root_id']for w in ww if w['stable4_physical'])),'short_run_only_roots':sorted(set(w['root_id']for w in ww)-set(w['root_id']for w in ww if w['stable4_physical'])),'complete_original_public_windows':sum(w['complete_original_public_window']for w in ww),'complete_original_public_roots':len(set(w['root_id']for w in ww if w['complete_original_public_window'])),'anchor_rgb_proxy_visible_windows':sum(w['rgb_proxy_anchor_visible']for w in ww),'all4_rgb_proxy_visible_windows':sum(w['rgb_proxy_all4_visible']for w in ww),'candidate_run_lengths':dict(Counter(w['hard_run_frames']for w in ww)),'environments':{env:{c:len(set(w['root_id']for w in ww if w['environment']==env and w['colour']==c))for c in COLORS}for env in ['A','B']}}
    save(E/'results/mining.json',{'summary':summary,'root_coverage':rootrows,'trajectories':trajectories,'candidate_windows':windows,'physical_endpoint_or_trace_label_mismatches':physical_mismatches,'C_or_D_trajectory_content_read':False,'input_reconstruction_blocker':'Full videos are lossy RGB and lack per-frame public instance masks. Private object poses/contacts must not be used to synthesize missing network inputs. Only explicitly saved public NPZ windows are directly reusable.','source_sha256':HASH})
    save(E/'results/supplement_timeline_private.json',full_motion);print('MINING_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
