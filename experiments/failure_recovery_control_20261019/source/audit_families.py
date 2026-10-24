from common import *
import argparse
def distance(a,b):
 a=np.asarray(a)[:,:3];b=np.asarray(b)[:,:3]
 return min(float(np.linalg.norm(a-b[list(p)],axis=1).max())for p in itertools.permutations(range(3)))
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',required=True,choices=['train_dev','evaluation']);a=pa.parse_args();guard('train'if a.phase=='train_dev'else'eval');known=read(S/'configs/AB_exclusion_inventory.json')['entries'];records=read(S/'data/data_manifest.json')['records'];roots=read(S/'data/roots_manifest.json');new=[];checks=[]
 for root in roots:
  rows=[r for r in records if r['root_id']==root['root_id']and r['branch_index']==0 and r['step']==0]
  if not rows:continue
  lab=read(rows[0]['labels_path']);poses=[lab['before']['continuous']['objects'][c]['position_m']for c in COLORS];comparators=[x for x in known+new if x['environment']==root['environment']];dd=[(distance(poses,x['poses']),x['source'])for x in comparators];best=min(dd);checks.append({'root_id':root['root_id'],'split':root['split'],'minimum_permutation_invariant_max_distance_m':best[0],'nearest_prior_source':best[1],'above_20mm':best[0]>.020});new.append({'environment':root['environment'],'poses':poses,'source':root['root_id']})
 save(S/f'results/families_{a.phase}.json',{'utc':now(),'passed':all(x['above_20mm']for x in checks),'threshold_m':.020,'actual_postsetup_positions':True,'color_permutations_checked':True,'known_AB_entries':len(known),'checks':checks});assert all(x['above_20mm']for x in checks),checks;print('FAMILIES_PASS',len(checks),min(x['minimum_permutation_invariant_max_distance_m']for x in checks),flush=True)
if __name__=='__main__':main()
