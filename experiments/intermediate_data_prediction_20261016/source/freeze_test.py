from config import *
def main():
 assert not(E/'results/collection_D3.json').exists();queries={};models={}
 for arm in ['old_coverage','new_coverage']:
  for seed in[0,1]:
   for typ,out in [('query',queries),('transition',models)]:
    name=f'{typ}_{arm}_seed{seed}';sel=read(E/'models'/name/'selected.json');log=read(E/'results'/f'train_{name}.json');assert sha(sel['path'])==sel['sha256'];assert log['updates']==(1200 if typ=='query' else 1000);best=min(log['log'],key=lambda x:x['score']);assert sel['path'].endswith(f'update{best["update"]:04d}.pt');out[name]=sel
 files=[Path(v['path'])for v in list(queries.values())+list(models.values())]+[E/'configs/protocol.json',E/'configs/roots.json',E/'configs/query_choices_frozen.json',E/'configs/query_training_frozen.json',E/'configs/transition_training_frozen.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt',REP/'source/labels_v2_1.py',REP/'source/model.py']+[E/'source'/f for f in ['config.py','predict.py','score.py','freeze_test.py']]
 save(E/'configs/before_D3_frozen.json',{'frozen_utc':now(),'query_heads':queries,'transition_models':models,'test_data_collected_or_seen':False,'thresholds':{'binary':.5,'drawer':'argmax'},'sha256':{str(p):sha(p)for p in files}});print('ALL_MODELS_FROZEN_BEFORE_D3',flush=True)
if __name__=='__main__':main()
