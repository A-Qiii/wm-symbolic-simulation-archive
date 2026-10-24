from pathlib import Path
import argparse,json,hashlib,gzip
from collections import Counter
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/core1200_v1';R=P/'runs/core1200_v1'
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--name',required=True);pa.add_argument('--videos',action='store_true');pa.add_argument('--include-test',action='store_true');args=pa.parse_args()
 if args.include_test:assert (R/'evaluation/model_test_freeze.json').exists(),'Test QA only after model freeze'
 if args.name=='full_train_dev_qa':
  from core1200_relabel import correct_manifest
  correct_manifest()
 m=json.loads((B/'data_manifest.json').read_text());rows=[x for x in m['records'] if args.include_test or x['split']!='test'];issues=[];stats=Counter();changes=Counter();zeros=0;replay=[];unknown=0;observed={};panels=[]
 allowed={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}
 for r in rows:
  d=Path(r['obs_before']).parent;l=json.loads(Path(r['labels_path']).read_text());res=json.loads((d/'result_private.json').read_text());acts=json.loads((d/'controls.json').read_text());stats[res['status']]+=1;stats['controls']+=len(acts)
  if len(acts)!=res['controls']:issues.append([str(d),'control_count'])
  before=l['before'];after=l['after'];valid=np.array(before['categorical_valid'])&np.array(after['categorical_valid']);diff=np.array(before['categorical'])!=np.array(after['categorical']);changes[r['skill']]+=int(np.sum(valid&diff));unknown+=int(np.sum(~np.array(after['categorical_valid'])))
  if not acts:
   zeros+=1
   if before!=after:issues.append([str(d),'zero_action_target_changed'])
  with np.load(r['obs_before']) as z:
   forbidden=set(z.files)-allowed-{k for k in z.files if k.startswith('calib_')}
   if forbidden:issues.append([str(d),'forbidden_public_keys',sorted(forbidden)])
   if any(not np.isfinite(z[k]).all() for k in ['depth_static','depth_gripper','robot_obs']):issues.append([str(d),'nonfinite_observation'])
   if len(panels)<12:
    pair=Image.new('RGB',(400,224),'white');pair.paste(Image.fromarray(z['rgb_static']).resize((200,200)),(0,24));pair.paste(Image.fromarray(z['rgb_gripper']).resize((200,200)),(200,24));ImageDraw.Draw(pair).text((5,5),f"{r['episode_id']} / {r['step']} {r['skill']}",fill='black');panels.append(pair)
  observed.setdefault(r['root_id'],set()).add(r['split'])
  if args.videos:
   for cam in ['rgb_static','rgb_gripper']:
    reader=imageio.get_reader(str(d/f'{cam}.mp4'));n=reader.count_frames();reader.close()
    if n!=len(acts)+1:issues.append([str(d),'video_frames',cam,n,len(acts)+1])
 for eid in {r['episode_id'] for r in rows}:
  x=json.loads((B/'episodes'/eid/'root_replay_check.json').read_text());replay.append(x)
  if not x['setup_all_successful']:issues.append([eid,'physical_setup_failed'])
 if any(len(x)>1 for x in observed.values()):issues.append(['root_leakage'])
 protocol=json.loads((R/'protocol.json').read_text());oldsha=hashlib.sha256((P/'data/data_manifest.json').read_bytes()).hexdigest()
 if oldsha!=protocol['old_data_manifest_sha256']:issues.append(['old_manifest_changed'])
 canvas=Image.new('RGB',(1200,224*((len(panels)+2)//3)),(232,236,242))
 for i,panel in enumerate(panels):canvas.paste(panel,((i%3)*400,(i//3)*224))
 canvas.save(R/f'{args.name}_contact_sheet.png')
 report={'passed':not issues,'transitions_checked':len(rows),'non_test_transitions_checked':sum(r['split']!='test' for r in rows),'root_count':len(observed),'stats':dict(stats),'zero_action_identical_endpoints':zeros,'categorical_changed_facts_by_skill':dict(changes),'invalid_after_facts':unknown,'issues':issues,'branch_replays':len(replay),'max_replay_scene_difference':max([x.get('scene_max_abs_difference',0) for x in replay],default=0),'max_replay_robot_difference':max([x.get('robot_max_abs_difference',0) for x in replay],default=0),'old_manifest_unchanged':oldsha==protocol['old_data_manifest_sha256'],'test_outcomes_inspected':args.include_test,'test_outcomes_used_for_model_selection':False,'video_frames_checked':args.videos,'contact_sheet_legend':'Each panel is actual pre-action RGB; static camera left, gripper camera right. Panel title gives root/branch, step and requested skill. No synthetic frames.'}
 (R/f'{args.name}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
