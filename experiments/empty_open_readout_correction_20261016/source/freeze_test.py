from config import *
def main():
 assert not(E/'results/collection_D4.json').exists();selected={}
 for arm in ['control','empty_open']:
  for seed in[0,1]:
   name=f'{arm}_seed{seed}';s=read(E/'models'/name/'selected.json');t=read(E/'results'/f'train_{name}.json');assert t['updates']==600 and t['draws']==19200;best=min(t['log'],key=lambda x:x['score']);assert s['path'].endswith(f'update{best["update"]:04d}.pt')and sha(s['path'])==s['sha256'];selected[name]=s
 for seed in[0,1]:assert read(E/'results'/f'train_control_seed{seed}.json')['initial_hash']==read(E/'results'/f'train_empty_open_seed{seed}.json')['initial_hash']
 initial=read(E/'configs/protocol.json')['query_training']['initial_heads'];files=[Path(x['path'])for x in list(selected.values())+list(initial.values())]+[E/'configs/protocol.json',E/'configs/roots.json',E/'configs/training_frozen.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt',PRIOR/'source/score.py',REP/'source/labels_v2_1.py']+[E/'source'/p for p in ['config.py','predict.py','score.py','freeze_test.py']];save(E/'configs/before_D4_frozen.json',{'frozen_utc':now(),'selected_heads':selected,'starting_heads':initial,'D4_collected_or_seen':False,'thresholds':{'binary':.5,'drawer':'argmax'},'sha256':{str(p):sha(p)for p in files}});print('MODELS_FIXED_BEFORE_D4',flush=True)
if __name__=='__main__':main()
