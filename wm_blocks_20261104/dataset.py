from common import *
import numpy as np
def specifications(mode):
 cfg=plan();out=[]
 if mode=='fit':
  out+=cfg['reused_fit']
  out +=[e for e in cfg['episodes']if e['split']in ['train','development']and not(e['split']=='development'and e['arm']=='covered')]
 else:out=[e for e in cfg['episodes']if e['split']==mode]
 return out
def raw(e):
 p=Path(e['source'])if 'source'in e else episode_path(e)
 if 'source_sha256'in e:assert sha(p)==e['source_sha256']
 return read(p),p
def summary(e):
 x,path=raw(e);L=min(e.get('length',len(e['actions'])),len(x['actions']));a=[r['action']for r in x['actions'][:L]];labels=x['labels'][:L+1]
 forbidden=[]
 for t,act in enumerate(a):
  if act==['Place','large','A'] and all(labels[t][n]['supported_by_large']is not False for n in NAMES[1:]):forbidden.append(t)
 return dict(id=e['id'],group=e['group'],split=e['split'],arm=e.get('arm','shared'),actions=a,length=L,source=str(path),source_sha256=sha(path),physical_valid=x['physical_valid'],forbidden=forbidden,complete=L==e.get('length',len(e['actions'])))
def audit(mode):
 cfg=plan();seeds={s:{e['seed']for e in cfg['episodes']if e['split']==s}for s in ['engineering','train','development','test']}
 for e in cfg['reused_fit']:seeds[e['split']].add(e['seed'])
 for i,s in enumerate(seeds):
  for t in list(seeds)[i+1:]:assert not(seeds[s]&seeds[t]),'Actual simulation seed overlap: '+s+'/'+t
 rows=[summary(e)for e in specifications(mode)];bad=[]
 for r in rows:
  if r['split']in ['train','development']and r['arm']!='covered'and r['forbidden']:bad.append(r['id'])
 assert not bad,'Actual held-out transition leaked: '+str(bad)
 sets={s:{r['group']for r in rows if r['split']==s}for s in ['train','development','test']}
 assert not(sets['train']&sets['development']or sets['train']&sets['test']or sets['development']&sets['test'])
 save(ROOT/f'reports/{mode}_data_audit.json',dict(episodes=len(rows),transitions=sum(r['length']for r in rows),groups={k:len(v)for k,v in sets.items()},invalid=[r['id']for r in rows if not r['physical_valid']],forbidden=[r for r in rows if r['forbidden']],rows=rows))
 return rows
def label_row(e,x,t):
 s=x['states'][t]['objects'];q=x['labels'][t]
 return dict(episode_id=e['id'],group=e['group'],split=e['split'],arm=e.get('arm','shared'),boundary=t,physical_valid=x['physical_valid'],xyz=[s[n]['position']for n in NAMES],shape=[[s[n]['aabb'][1][j]-s[n]['aabb'][0][j]for j in range(3)]for n in NAMES],region=[q[n]['region']for n in NAMES],held=[-1 if q[n]['held']is None else int(q[n]['held'])for n in NAMES])
