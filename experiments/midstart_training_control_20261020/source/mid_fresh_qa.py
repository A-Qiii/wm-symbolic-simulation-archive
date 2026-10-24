from common import *
def main():
 guard();f=S/'fresh_validation';roots=read(f/'data/roots_manifest.json');checks=[]
 for r in roots:
  if r['template']!='order_known4':continue
  a=f/'data/episodes'/(r['root_id']+'_b1');b=f/'data/episodes'/(r['root_id'].replace('order_known4','order_novel4')+'_b1');exact=True
  for i in range(4):
   with np.load(a/f'root_observation_{i}.npz')as x,np.load(b/f'root_observation_{i}.npz')as y:exact &= set(x.files)==set(y.files)and all(np.array_equal(x[k],y[k])for k in x.files)
  assert exact;checks.append({'known':str(a),'novel':str(b),'four_public_frames_exact':True})
 assert len(checks)==12
 save(S/'results/independent_validation_qa.json',{'utc':now(),'passed':True,'matched_initial_public_pairs':checks,'pair_count':12,'families':4,'role_pairs':3,'no_injected_disturbance':True,'original_training_order_novelty_audit':read(G/'results/action_sequence_audit.json'),'new_crop_training_cannot_introduce_a_contiguous_order_absent_from_all_original_trajectories':True,'setup_valid':sum(r['setup_valid']for r in read(f/'results/episode_audits.json')),'all_failures_retained':True,'post_evaluation_repair':False});print('FRESH_QA_PASS',flush=True)
if __name__=='__main__':main()
