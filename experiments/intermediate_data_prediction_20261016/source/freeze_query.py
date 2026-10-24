from config import *
def main():
 assert not list((E/'models').glob('*'));files=list((E/'data/query_train').glob('*.npz'))+list((E/'data/query_evaluator').glob('*.npz'))+list((E/'configs').glob('query_*.npz'))+[E/'configs/query_inputs.json',E/'configs/protocol.json',E/'configs/roots.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt',REP/'source/model.py']+[E/'source'/x for x in ['config.py','prepare_query.py','train_query.py','freeze_query.py']]
 for seed in [0,1]:
  aa=np.load(E/'configs'/f'query_schedule_old_coverage_seed{seed}.npz');bb=np.load(E/'configs'/f'query_schedule_new_coverage_seed{seed}.npz');assert np.array_equal(aa['state_ids'][:,:20],bb['state_ids'][:,:20]);assert aa['indices'].shape==bb['indices'].shape==(1200,32)
 assert not(E/'results/collection_D3.json').exists();save(E/'configs/query_training_frozen.json',{'frozen_utc':now(),'D3_collected':False,'same20_base_draws':True,'sha256':{str(p):sha(p)for p in files}});print('QUERY_TRAINING_FROZEN',len(files),flush=True)
if __name__=='__main__':main()
