from config import *
def main():
 assert not list((E/'models').glob('transition_*'));assert not(E/'results/collection_D3.json').exists()
 files=list((E/'data/transition_train').glob('*'))+list((E/'configs').glob('transition_schedule*.npz'))+[E/'configs/query_choices_frozen.json',E/'configs/protocol.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt',REP/'source/model.py',REP/'source/runtime.py']+[E/'source'/f for f in ['config.py','prepare_transition.py','train_transition.py','freeze_transition.py']]
 for seed in[0,1]:
  sc=np.load(E/'configs'/f'transition_schedule_seed{seed}.npz');assert sc['base'].shape==(1000,48)and sc['new_aux'].shape==sc['old_aux'].shape==(1000,16)
 save(E/'configs/transition_training_frozen.json',{'frozen_utc':now(),'D3_collected':False,'sha256':{str(p):sha(p)for p in files}});print('TRANSITION_TRAINING_FROZEN',flush=True)
if __name__=='__main__':main()
