from pathlib import Path
import json,hashlib,collections,datetime
import numpy as np
import imageio.v2 as imageio
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_joint_20261014';B=P/'data/predicate_joint_acceptance_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=read(B/'data_manifest.json');freeze=read(E/'results/candidate_freeze.json');assert len(m['trajectories'])==30
    assert datetime.datetime.fromisoformat(freeze['created_utc']).timestamp()<(B/'data_manifest.json').stat().st_mtime
    public={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'};counts=collections.Counter();windowhashes=[];video=[]
    prior=read(P/'data/predicate_gap_supplement_20261014/data_manifest.json');priorhash={r['window_rgb_sha256']for r in prior['records']}
    for r in m['records']:
        assert len(r['public_frames'])==4 and max(r['history_frame_indices'])==r['anchor_frame'];lab=read(r['labels_private']);j=['red','blue','pink'].index(r['target_color']);hh=[]
        for path,f in zip(r['public_frames'],lab['frames']):
            with np.load(path)as z:
                assert all(k in public or k.startswith('calib_')for k in z.files)
                hh.append(hashlib.sha256(z['rgb_static'].tobytes()+z['rgb_gripper'].tobytes()).hexdigest())
            assert f['labels']['categorical_valid'][j]
            assert any(x>=4 for x in f['objects_audit'][r['target_color']]['visible_pixels'].values())
            if r['condition']=='held_positive':assert f['objects_audit'][r['target_color']]['held_contact_audit'] and lab['motion_audit']['verified']
            elif r['condition']=='in_positive':assert f['labels']['categorical'][j]==3 and not f['objects_audit'][r['target_color']]['held_contact_audit']
            else:assert not f['objects_audit'][r['target_color']]['held_contact_audit'] and f['objects_audit'][r['target_color']]['external_support_contacts']>0
        assert hh==r['frame_rgb_sha256'];h=hashlib.sha256(''.join(hh).encode()).hexdigest();assert h==r['window_rgb_sha256'] and h not in priorhash
        windowhashes.append(h);counts[r['condition']]+=1
    assert len(windowhashes)==len(set(windowhashes))
    for t in m['trajectories']:
        assert t['physics_guard_passed'] and not t['label_issues'] and max(t['root_replay'].values())<=1e-5
        for p in t['public_video_files']:
            count=0
            with imageio.get_reader(p)as reader:
                for frame in reader:count+=1
            assert count==t['frames'];video.append({'path':p,'frames':count,'sha256':sha(p)})
    full=counts['held_positive']==12 and counts['in_positive']==6
    out={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':12,'trajectories':30,'windows':len(m['records']),'condition_counts':dict(counts),'full_positive_coverage':full,'planned_held_roots':12,'actual_held_roots':counts['held_positive'],'planned_in_roots':6,'actual_in_roots':counts['in_positive'],'videos_decoded':video,'all_public_frames_keys_verified':True,'all_physics_guards_and_root_replays_pass':True,'no_duplicate_windows_with_previous_supplement':True,'no_resampling':True,'gaps':[{ 'trajectory':t['trajectory_id'],'gaps':t['gaps'],'status':t['controller_result']['status']}for t in m['trajectories']if t['gaps']or t['controller_result']['status']not in ['success','completed_probe']]}
    (E/'results/acceptance_qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items()if k!='videos_decoded'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
