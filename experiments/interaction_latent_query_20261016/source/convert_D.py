from common import *
from features import Encoder
from representation import transform
import fcntl
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base=E/'data/D';raw=E/'data/D_raw';(base/'public/features').mkdir(parents=True,exist_ok=True);(base/'_private').mkdir(exist_ok=True);rows=[];labels={};pairs=[];outcomes=[];wi={};roots={r['root_id']:r for r in read(raw/'roots_manifest.json')}
 for ep in read(raw/'manifest.json')['episodes']:
  folder=raw/'episodes'/ep['episode_id'];wins=read(folder/'_private/windows.json')
  for w in wins:
   i=len(rows);sid=f'd{i:05d}';path=w['public_frames'][-1];rows.append({'id':sid,'index':i,'root_id':w['root_id'],'episode_id':w['episode_id'],'environment':'D','split':'test','path':path,'sha256':sha(path),'condition':w['condition'],'anchor':w['anchor']});wi[(ep['episode_id'],w['anchor'])]=i;labels[str(i)]={'y':w['new_labels']['y'],'v':w['new_labels']['v']}
  for r in read(folder/'_private/actions.json'):
   rid=ep['root_id'];target=COLORS.index(roots[rid]['color']);pid=f'p{len(pairs):05d}';pairs.append({'id':pid,'root_id':rid,'episode_id':ep['episode_id'],'step':r['step'],'skill':r['skill'],'action':action(r['skill'],target,96),'before':wi[(ep['episode_id'],r['start'])],'after':wi[(ep['episode_id'],r['end'])],'split':'test','dataset':'freshD','condition':ep['condition']});outcomes.append({'pair_id':pid,'status':r['native']['status'],'normal_executable':r['native']['status']=='success'})
 enc=Encoder();fs=[]
 for k in range(0,len(rows),8):
  out=enc.encode([r['path']for r in rows[k:k+8]]);fs+=out
  for r,f in zip(rows[k:k+8],out):np.savez_compressed(base/'public/features'/f"{r['id']}.npz",**f)
 f={k:np.stack([x[k]for x in fs])for k in fs[0]};norm=dict(np.load(E/'data/public/normalization.npz'));obj,scene=transform(f,norm);np.savez_compressed(base/'public/inputs.npz',objects=obj,scene=scene)
 chains=[]
 for p in pairs:
  c=[p['id']];cur=p
  for _ in range(2):
   nxt=[q for q in pairs if q['episode_id']==cur['episode_id']and q['step']==cur['step']+1 and q['before']==cur['after']]
   if not nxt:break
   assert len(nxt)==1;cur=nxt[0];c.append(cur['id'])
  chains.append(c)
 save(base/'public/states.json',rows);save(base/'public/pairs.json',pairs);save(base/'public/chains.json',chains);save(base/'public/roots.json',list(roots.values()));save(base/'_private/labels.json',labels);save(base/'_private/pair_outcomes.json',outcomes);save(E/'results/D_data_audit.json',{'roots':len(roots),'episodes':len(read(raw/'manifest.json')['episodes']),'states':len(rows),'pairs':len(pairs),'native_statuses':{s:sum(x['status']==s for x in outcomes)for s in['success','refused','failed']},'D_fit_normalization_or_weights':False,'old_D_consumed':False,'observed_three_step_segments':sum(len(c)==3 for c in chains),'storage_bytes':sum(p.stat().st_size for p in raw.rglob('*')if p.is_file())});print('D_CONVERTED',len(rows),len(pairs),flush=True)
if __name__=='__main__':main()
