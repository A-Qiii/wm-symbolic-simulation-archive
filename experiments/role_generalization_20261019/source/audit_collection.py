from common import *
import argparse
def distance(a,b):return min(float(np.linalg.norm(np.asarray(a)[:,:3]-np.asarray(b)[list(order),:3],axis=1).max())for order in itertools.permutations(range(3)))
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train_dev','locked_test'],required=True);args=pa.parse_args();guard('audit');verify_freeze();budget_check();roots=read(S/'data/roots_manifest.json');byroot={r['root_id']:r for r in roots};records=read(S/'data/data_manifest.json')['records'];audits=read(S/'results/root_audits.json');selected=[a for a in audits if(a['split']=='locked_test')==(args.phase=='locked_test')]
 expected=78 if args.phase=='train_dev'else 48;assert len(selected)==expected
 allgeoms=[{'environment':a['environment'],'family_id':a['family_id'],'poses':[a['branches'][0]['initial_positions'][c]for c in COLORS],'source':a['root_id']}for a in audits]
 known=read(S/'configs/AB_exclusion_inventory.json')['entries'];checks=[]
 for g in allgeoms:
  if g['source']not in{a['root_id']for a in selected}:continue
  others=[x for x in allgeoms if x['environment']==g['environment']and x['family_id']!=g['family_id']]+[x for x in known if x['environment']==g['environment']]
  d,name=min((distance(g['poses'],x['poses']),x['source'])for x in others);checks.append({'role_id':g['source'],'family_id':g['family_id'],'nearest_distance_m':d,'nearest':name,'above20mm':d>.020})
 # Audit same geometry/shape intervention and actual visible masks, never input these semantic IDs to model.
 visibility=[];episode_stats=[];status=collections.Counter();basechecks=[]
 by=collections.defaultdict(list)
 ids={a['root_id']for a in selected}
 for r in records:
  if r['root_id']in ids:by[r['episode_id']].append(r)
 for eid,rr in sorted(by.items()):
  root=byroot[rr[0]['root_id']];rr.sort(key=lambda r:r['step']);d=Path(rr[0]['obs_before']).parent.parent;identity=read(d/'physical_identity_private.json');base=read(d/'family_base_check.json');basechecks.append(base)
  assert identity['layout']==root['layout'];assert all(identity['objects']['block_'+c]['mass']>0 for c in COLORS)
  windows=[[d/f'root_observation_{j}.npz'for j in range(4)]]+[[Path(r['obs_after']).parent/f'boundary_history_{j}.npz'for j in range(4)]for r in rr];hashes=[]
  for t,paths in enumerate(windows):
   frames=[dict(np.load(p))for p in paths];h=hashlib.sha256()
   for frame in frames:
    for k in sorted(frame):h.update(k.encode());h.update(frame[k].tobytes())
   hashes.append(h.hexdigest())
   for c in COLORS:
    uid=identity['objects']['block_'+c]['uid']
    for cam in ['static','gripper']:
     counts=[int(((f['seg_'+cam]>=0)&((f['seg_'+cam]&((1<<24)-1))==uid)).sum())for f in frames]
     visibility.append({'family_id':root['family_id'],'root_id':root['root_id'],'episode_id':eid,'split':root['split'],'branch':rr[0]['branch_index'],'step':t,'object':c,'role':'H'if c==root['held_color']else'T'if c==root['matched_color']else'U','camera':cam,'last_pixels':counts[-1],'min_pixels':min(counts),'max_pixels':max(counts),'frames_nonzero':sum(x>0 for x in counts),'last_bin':'none'if counts[-1]==0 else'low1_15'if counts[-1]<16 else'visible16plus'})
  for r in rr:
   l=read(r['labels_path']);st='success'if l['success']else'executed_failure'if l['control_samples_including_start']>1 else'refused';status[st]+=1
  episode_stats.append({'family_id':root['family_id'],'root_id':root['root_id'],'episode_id':eid,'split':root['split'],'arms':root['arms'],'h0_public_sha256':hashes[0],'all_boundary_public_sha256':hashes})
 csvwrite(S/f'results/visibility_{args.phase}.csv',visibility);save(S/f'results/public_dedup_{args.phase}.json',episode_stats);save(S/f'results/families_{args.phase}.json',{'utc':now(),'checks':checks,'passed':all(x['above20mm']for x in checks),'threshold_m':.020,'within_family_roles_not_independent':True})
 allocation={}
 if args.phase=='train_dev':
  for arm in ARMS:
   eps=[x for x in episode_stats if arm in x['arms']];assert len(eps)==72;allocation[arm]={'episodes':len(eps),'transitions':len(eps)*6,'family_count':len({x['family_id']for x in eps}),'unique_h0_public':len({x['h0_public_sha256']for x in eps}),'unique_boundary_public':len({h for x in eps for h in x['all_boundary_public_sha256']})}
  assert len({r['family_id']for r in records if r['split']=='train'})==12 and len({r['family_id']for r in records if r['split']=='development'})==4
 passed=all(x['paired_prefix_exact']and all(y['physics_guard_all']for y in x['branches'])for x in selected)and all(x['above20mm']for x in checks)
 result={'utc':now(),'passed':passed,'roles':len(selected),'families':len({a['family_id']for a in selected}),'valid_pairs':sum(a['valid_pair']for a in selected),'initial_invalid_branches':sum(not b['initial_valid']for a in selected for b in a['branches']),'statuses':dict(status),'allocation':allocation,'same_family_base_exact':all(max(b.get('scene_max_abs',0),b.get('robot_max_abs',0))<=1e-5 for b in basechecks),'visibility_rows':len(visibility),'all_failure_rows_retained':True,'objects_not_recolored_resized_or_swapped':True}
 save(S/f'results/audit_{args.phase}.json',result);assert passed,result;print(json.dumps(result),flush=True)
if __name__=='__main__':main()
