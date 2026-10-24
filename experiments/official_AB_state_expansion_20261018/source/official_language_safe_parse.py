"""Interpret pickle opcodes as inert data only; never deserialize Python objects."""
from pathlib import Path
import json,pickletools,collections,hashlib,shutil
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_AB_state_expansion_20261018';OLD=P/'experiments/official_AB_state_pilot_20261018'
ALLOWED={'numpy.core.multiarray _reconstruct','numpy ndarray','numpy dtype','numpy.core.multiarray scalar'}
class Node:
 def __init__(self,constructor,args):self.constructor=constructor;self.args=args;self.state=None
def read(p):return json.loads(Path(p).read_text())
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def parse(body):
 stack=[];memo={};mark=object();count=0
 def marked():
  n=len(stack)-1
  while n>=0 and stack[n]is not mark:n-=1
  assert n>=0
  values=stack[n+1:];del stack[n:];return values
 for op,arg,pos in pickletools.genops(body):
  count+=1;n=op.name
  if n=='PROTO':assert arg<=4
  elif n=='GLOBAL':assert arg in ALLOWED;stack.append(('INERT_GLOBAL',arg))
  elif n in ['BINPUT','LONG_BINPUT']:assert arg not in memo;memo[arg]=stack[-1]
  elif n in ['BINGET','LONG_BINGET']:stack.append(memo[arg])
  elif n in ['BININT1','BININT2','BININT','LONG1','LONG4','BINUNICODE','SHORT_BINUNICODE','SHORT_BINBYTES','BINBYTES','BINFLOAT']:stack.append(arg)
  elif n=='NEWFALSE':stack.append(False)
  elif n=='NEWTRUE':stack.append(True)
  elif n=='NONE':stack.append(None)
  elif n=='EMPTY_TUPLE':stack.append(())
  elif n=='EMPTY_LIST':stack.append([])
  elif n=='EMPTY_DICT':stack.append({})
  elif n=='MARK':stack.append(mark)
  elif n=='TUPLE':stack.append(tuple(marked()))
  elif n in ['TUPLE1','TUPLE2','TUPLE3']:
   k=int(n[-1]);value=tuple(stack[-k:]);del stack[-k:];stack.append(value)
  elif n=='REDUCE':
   args=stack.pop();constructor=stack.pop();assert isinstance(args,tuple)and constructor[0]=='INERT_GLOBAL';stack.append(Node(constructor[1],args))
  elif n=='BUILD':
   state=stack.pop();assert isinstance(stack[-1],Node)and stack[-1].state is None;stack[-1].state=state
  elif n=='APPEND':
   value=stack.pop();assert type(stack[-1])is list;stack[-1].append(value)
  elif n=='APPENDS':
   values=marked();assert type(stack[-1])is list;stack[-1].extend(values)
  elif n=='SETITEMS':
   values=marked();assert type(stack[-1])is dict and len(values)%2==0
   for key,value in zip(values[::2],values[1::2]):assert type(key)is str and key not in stack[-1];stack[-1][key]=value
  elif n=='STOP':assert len(stack)==1 and pos==len(body)-1;return stack[0],{'opcodes':count,'memo_entries':len(memo),'executed_input_callables':0}
  else:raise ValueError(('Unsupported inert opcode',n,pos))
 raise ValueError('No STOP')
def scalar_int(o):
 if type(o)is int:return o
 assert isinstance(o,Node)and o.constructor=='numpy.core.multiarray scalar'and len(o.args)==2
 dtype,data=o.args;assert isinstance(dtype,Node)and dtype.constructor=='numpy dtype'and dtype.args[0]=='i8'
 assert dtype.state[1]in ['<','=']and isinstance(data,bytes)and len(data)==8
 return int.from_bytes(data,'little',signed=True)
def main():
 path=E/'data/metadata/auto_lang_ann.npy'
 with path.open('rb')as f:
  v=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,v);body=f.read()
 assert dtype.hasobject and shape==();root,audit=parse(body)
 assert root.constructor=='numpy.core.multiarray _reconstruct';assert root.state[1]==()and len(root.state[4])==1
 data=root.state[4][0];assert set(data)=={'language','info'}
 language,info=data['language'],data['info'];assert set(language)=={'ann','task','emb'};assert set(info)=={'episodes','indx'}or set(info)=={'indx'}
 ann,task,indices=language['ann'],language['task'],info['indx'];assert type(ann)is list and type(task)is list and type(indices)is list and len(ann)==len(task)==len(indices)
 assert all(type(s)is str for s in ann+task)
 segments=[];episodes=np.load(OLD/'data/metadata/training/ep_start_end_ids.npy',allow_pickle=False);mapping=read(OLD/'configs/verified_scene_ranges.json')['ranges'];counts=collections.Counter();cross=[]
 for i,(a,t,pair)in enumerate(zip(ann,task,indices)):
  assert type(pair)is tuple and len(pair)==2;s,e=map(scalar_int,pair);assert 0<=s<=e
  scenes=[c for c,(lo,hi)in mapping.items()if lo<=s<=e<=hi];roots=[k for k,(lo,hi)in enumerate(episodes)if lo<=s<=e<=hi]
  if len(scenes)!=1 or len(roots)!=1:cross.append({'annotation_index':i,'interval':[s,e],'scenes':scenes,'episode_indices':roots});continue
  scene=scenes[0];counts[scene]+=1
  if scene not in ['A','B']:continue
  k=roots[0];segments.append({'annotation_index':i,'scene':scene,'official_episode_index':k,'official_episode_inclusive':[int(x)for x in episodes[k]],'root_id':f'official_ABC_{scene}_episode_{k:04d}','task':t,'annotation':a,'interval_inclusive':[s,e],'semantics':'candidate hint only; never label truth'})
 emb=language['emb'];assert isinstance(emb,Node)
 save(E/'results/task_segments_A_B.json',segments)
 summary={'parser_audit':audit,'metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'annotation_count':len(ann),'scenes_annotation_counts':dict(counts),'rejected_cross_boundary_annotations':cross,'AB_segments':len(segments),'AB_source_episode_counts':{s:len({x['root_id']for x in segments if x['scene']==s})for s in ['A','B']},'task_counts_AB':dict(collections.Counter(x['task']for x in segments)),'embedding_ignored':True,'embedding_shape_as_inert_metadata':list(emb.state[1]),'safe_integer_rule':'Only numpy int64 scalar byte payload is decoded as literal little-endian signed integer; no dtype or scalar constructor is invoked','root_keys':list(data),'language_keys':list(language),'info_keys':list(info),'no_C_D_frames_selected':True}
 save(E/'results/metadata_safe_summary.json',summary);shutil.copy2(__file__,E/'source'/Path(__file__).name)
 print(json.dumps({k:v for k,v in summary.items()if k not in['rejected_cross_boundary_annotations']},ensure_ascii=False));print('CROSS_BOUNDARY_COUNT',len(cross))
if __name__=='__main__':main()
