from common import *
import subprocess,sys,time,torch

def run(name,size,*args):
 env=os.environ.copy();env['RSSM_SCALE_ROOTS']=str(size)
 print('START',name,size,args,now(),flush=True)
 with (B/'logs'/('finish_'+name+'_'+str(size)+'_'+'_'.join(args)+'.log')).open('a')as log:
  subprocess.run([sys.executable,str(B/'source'/name),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,size,args,now(),flush=True)

def main():
 frozen=read(B/'configs/all_models_frozen.json');assert len(frozen['sha256'])==8
 for p,h in frozen['sha256'].items():assert sha(p)==h
 assert not(B/'results/compute_complete.json').exists()
 for size in [20,40]:run('evaluate.py',size,'train')
 for size in [20,40]:run('probes.py',size,'train')
 for seed in [0,1]:
  qs=[np.load(B/f'cells/roots{s}/data/probe_schedule_seed{seed}.npy')for s in [20,40]];assert np.array_equal(*qs)
  hashes=[]
  for size in [20,40]:
   for arm in ARMS:
    rec=read(B/f'cells/roots{size}/results/probe_train_{arm}_seed{seed}.json');assert rec['updates']==600 and rec['train_roots']==20 and rec['train_records']==300;hashes.append(rec['initial_hash'])
  assert len(set(hashes))==1
 paths=list((B/'cells').glob('roots*/models/probe_*.pt'));assert len(paths)==8
 if (B/'configs/all_probes_frozen.json').exists():
  for p,h in read(B/'configs/all_probes_frozen.json')['sha256'].items():assert sha(p)==h
 else:save(B/'configs/all_probes_frozen.json',{'utc':now(),'fits':8,'updates_each':600,'all_mainmodels_frozen':True,'before_ABtest_orC_predictions':True,'sha256':{str(p):sha(p)for p in paths}})
 for size in [20,40]:run('evaluate.py',size,'heldout')
 for size in [20,40]:run('probes.py',size,'predict')
 for p,h in frozen['sha256'].items():assert sha(p)==h
 save(B/'results/compute_complete.json',{'utc':now(),'main_fits':8,'main_updates':48000,'probe_fits':8,'probe_updates':4800,'heldout_after_allfits_frozen':True,'shared_probe_schedule_andinitialization_verified':True,'no_D':True,'no_new_collection':True})
 print('ALL_TRAINING_AND_PREDICTIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
