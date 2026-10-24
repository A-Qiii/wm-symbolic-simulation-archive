"""Post-hoc entity registration audit, with private UIDs used only for checking."""
from pathlib import Path
import json,hashlib
import numpy as np
from wm_symbolic.vision_cache import VisibleRegistry
E=Path('/home/robot/research/projects/wm_symbolic/experiments/state_predicates_joint_20261014')
def main():
    meta=json.loads((E/'data/metadata_acceptance.json').read_text());reg=None;last=None;rows=[]
    for r in meta:
        if r['trajectory_id']!=last:reg=VisibleRegistry();last=r['trajectory_id']
        frames=[]
        for path in r['public_frames']:
            with np.load(path)as z:frames.append({k:z[k]for k in ['rgb_static','rgb_gripper','seg_static','seg_gripper']})
        reg.update(frames);lab=json.loads(Path(r['labels_private']).read_text())['frames'][-1]
        for j,c in enumerate(['red','blue','pink']):
            actual=lab['objects_audit'][c]['object_uid_evaluator_only'];found=reg.colour_ids.get(j);rows.append({'sample_id':r['sample_id'],'colour':c,'pixel_registered_uid':found,'evaluator_uid':actual,'match':found==actual})
    out={'windows_checked':len(meta),'entity_matches':sum(r['match']for r in rows),'mismatches':[r for r in rows if not r['match']],'rows':rows,'diagnostic_only_private_uid_not_used_by_model':True,'no_model_forward':True,'metadata_sha256':hashlib.sha256((E/'data/metadata_acceptance.json').read_bytes()).hexdigest()}
    assert not out['mismatches'];(E/'results/registry_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('REGISTRY_AUDIT_SAVED',out['entity_matches'])
if __name__=='__main__':main()
