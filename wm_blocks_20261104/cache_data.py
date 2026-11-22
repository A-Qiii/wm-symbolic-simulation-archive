from common import *
from dataset import specifications,raw,label_row,audit
import numpy as np,torch,argparse
from rgb_front_three import RGBFront
def main(mode):
 out=ROOT/'cache'/mode
 if(out/'complete.json').exists():return
 audit(mode);out.mkdir(exist_ok=True,parents=True);rows=[];values=[];front=RGBFront();st=time.monotonic();torch.cuda.reset_peak_memory_stats();reused=0
 try:
  for e in specifications(mode):
   x,_=raw(e);L=min(e.get('length',len(e['actions'])),len(x['actions']))
   for t in range(L+1):
    dest=out/(e['id']+f'_b{t}.npz')
    if dest.exists():f=np.load(dest)['features']
    else:
     legacy=Path(e['cache_source_dir'])/(e['cache_source_id']+f'_b{t}.npz')if 'cache_source_dir'in e else Path('/__no_cache__')
     if legacy.exists():
      assert e['extractor_sha256']==sha(ROOT/'src/rgb_front_three.py');f=np.load(legacy)['features'];reused+=1
     else:f,_=front.extract(x['windows'][t])
     np.savez_compressed(dest,features=f)
    assert f.shape==(3,1567)and np.isfinite(f).all();rows.append(label_row(e,x,t));values.append(f)
   status('cache_'+mode,episodes=len({r['episode_id']for r in rows}),boundaries=len(rows),latest=e['id'])
  np.savez_compressed(out/'all_features.npz',features=np.stack(values));save(out/'rows.json',rows);save(out/'complete.json',dict(boundaries=len(rows),reused_frozen_features=reused,seconds=time.monotonic()-st,peak_cuda_bytes=torch.cuda.max_memory_allocated(),features_source_sha256=sha(ROOT/'src/rgb_front_three.py')))
 finally:front.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','engineering','test']);a=p.parse_args();gpu=lock();main(a.mode)
