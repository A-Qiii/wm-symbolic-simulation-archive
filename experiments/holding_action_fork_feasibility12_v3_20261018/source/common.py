from pathlib import Path
import json,hashlib,datetime,base64,os,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');NAME='holding_action_fork_feasibility12_v3_20261018';E=P/'experiments'/NAME;R=P/'reports'/NAME
def read(p):return json.loads(Path(p).read_text())
def json_default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):
  value={"raw_bytes_encoding":"base64","data":base64.b64encode(x).decode("ascii")}
  try:value["utf8_text"]=x.decode("utf-8",errors="strict")
  except UnicodeDecodeError:value["utf8_text"]=None
  return value
 return str(x)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=json_default)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def mesh(path):
 s=Path(path).read_text().splitlines();v=np.array([[float(x)for x in a.split()[1:]]for a in s if a.startswith('v ')]);f=np.array([[int(x.split('/')[0])-1 for x in a.split()[1:]]for a in s if a.startswith('f ')]);assert f.shape[1]==3;return v,f
def bbox(mask):
 y,x=np.where(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]if len(x)else None

def save_raw_npz(path,**arrays):
 path=Path(path);tmp=path.with_suffix('.partial')
 with tmp.open('wb')as stream:
  np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
 tmp.replace(path)
 with np.load(path,allow_pickle=False)as z:
  assert set(z.files)==set(arrays)
  for k,v in arrays.items():assert np.array_equal(z[k],v,equal_nan=np.issubdtype(np.asarray(v).dtype,np.inexact)),k


COLORS=['red','blue','pink']
B=P/'experiments/relation_GT_boxes36_ABC_20261018'

OLD=P/'experiments/holding_action_fork_feasibility12_20261018'
OLDROOT=OLD/'data/private/fork12_A_00'

Q=E/'qualification'
DIAG=P/'experiments/native_restore_semantics_root1_final_20261018'

import copy
def split_state(x):
 """Preserve all fields; classify prior solver outputs separately, never discard."""
 x=json.loads(json.dumps(x,default=json_default));reported=[]
 for body in x['bodies']:
  for j,st in enumerate(body['joint_states']):
   st=list(st);reported.append({'body':body['uid'],'joint':j,'applied_motor_effort':st[3]});st[3]='prior_step_reported_motor_effort';body['joint_states'][j]=st
 for c in x['constraints']:reported.append({'constraint':c['id'],'reported_applied_forces':c['applied_forces']});c['applied_forces']='prior_step_reported_constraint_force'
 return x,reported

def archive_renderer_schema(x):
 x=copy.deepcopy(x);found=[]
 for body in x['bodies']:
  for i,shape in enumerate(body['visual_shapes']):
   if body['uid']==0 and shape[1]==12:
    assert shape[2] not in range(1,21),'Expected proven EGL cache-metadata defect; do not normalize valid geometry'
    found.append({'body':0,'link':12,'index':i,'raw_geometry_type':shape[2],'raw_dimensions':shape[3],'raw_filename':shape[4]})
    shape[2]='undefined_EGL_cache_metadata_geometry_type';shape[3]='undefined_EGL_cache_metadata_dimensions';shape[4]='undefined_EGL_cache_metadata_filename'
 assert len(found)==1
 return x,found

