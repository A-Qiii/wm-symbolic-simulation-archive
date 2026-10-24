from coverage_common import *
import shutil
def main():
 assert read(E/'results/encoding_parity.json')['passed'];assert not(E/'configs/training_frozen.json').exists();rows=read(E/'data/public/states.json');labs=np.load(E/'data/private/official_labels.npz',allow_pickle=False);z=np.load(E/'data/public/encoded.npz',allow_pickle=False);assert np.array_equal(z['state_ids'],labs['state_ids'])and np.array_equal(z['root_ids'],labs['root_ids'])
 for split,name in [('train_candidate','official_train'),('validation_candidate','official_validation')]:
  ix=np.flatnonzero(labs['split']==split);assert len(ix)==120;data={k:labs[k][ix]for k in['y','v','root_ids','state_ids']};data['z']=z['z'][ix];dest=E/'data'/('train'if split=='train_candidate'else'evaluation')/f'{name}.npz';np.savez_compressed(dest,**data)
 for head in HEADS:
  d=np.load(E/'data/train'/f'old_{head}.npz',allow_pickle=False);assert set(d['root_ids']).isdisjoint(labs['root_ids'])
  for q in(E/'data/evaluation').glob('*.npz'):assert set(d['root_ids']).isdisjoint(np.load(q,allow_pickle=False)['root_ids'])
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 trainfiles=list((E/'data/train').glob('*'))+list((E/'configs').glob('normalization*.npz'))+list((E/'configs').glob('schedule*.npz'))+list((E/'models').glob('initial*.pt'))+[E/'configs/protocol.json']+list((E/'source').glob('*.py'))
 save(E/'configs/training_frozen.json',{'utc':now(),'before_any_optimizer_update':True,'sha256':{str(p):sha(p)for p in trainfiles},'max_runs':8,'updates_each':600,'all_official_train_states_used':120,'validation_file_hashes':{str(p):sha(p)for p in(E/'data/evaluation').glob('*.npz')}})
 print('TRAIN_INPUTS_FROZEN',flush=True)
if __name__=='__main__':main()
