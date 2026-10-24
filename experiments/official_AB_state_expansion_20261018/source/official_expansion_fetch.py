"""Bounded official A/B expansion retrieval. Untrusted pickle is never executed."""
from pathlib import Path
import json,struct,zlib,hashlib,shutil,datetime,pickletools,io,collections
import numpy as np
import official_ab_fetch as F
P=F.P;OLD=P/'experiments/official_AB_state_pilot_20261018'
E=P/'experiments/official_AB_state_expansion_20261018';R=P/'reports/official_AB_state_expansion_20261018'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):F.write(Path(p),v)
def initialize():
 assert not E.exists(),'Fresh expansion only; never overwrite existing run'
 for sub in ['source','configs','cache','data/metadata','data/frames','results','logs']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True,exist_ok=False)
 prior=read(OLD/'results/network_ledger.json');assert prior['body_bytes_read']==268218222
 for p in [Path(__file__),Path(F.__file__)]:shutil.copy2(p,E/'source'/p.name)
 write(E/'results/network_ledger.json',{'body_bytes_read':prior['body_bytes_read'],'base_body_bytes_read':prior['body_bytes_read'],'base_ledger':str(OLD/'results/network_ledger.json'),'base_ledger_sha256':F.sha(OLD/'results/network_ledger.json'),'requests':[]})
 write(E/'configs/discovery_protocol.json',{'utc':F.now(),'sources':'official training A/B only; metadata includes ABC but only A/B segments may select frames; no C/D source frames','prior_body_bytes':prior['body_bytes_read'],'cumulative_cap_bytes':F.BUDGET,'max_new_source_episodes':8,'new_episode_cap_per_scene':4,'max_new_source_frames':256,'max_new_bounded_simulation_runs_including_old_source_retests':256,'metadata_scope':'one standard training/lang_annotations/auto_lang_ann.npy only, checked against retained central directory','untrusted_pickle':'Never pickle.load, numpy.load allow_pickle=True, imports or callable invocation from input. Use pickletools symbolic opcode parsing and inert constructor records. Reject unsupported structures.','no_training':True,'no_model_based_sampling':True,'final_frame_selection_and_episode_split_frozen_before_new_frame_download_or_simulation':True})
 F.E=E
def extract(meta,out,metadata=False):
 F.E=E;out=Path(out);assert not out.exists()
 cap=64*1024**2 if metadata else 16*1024**2
 assert meta['uncompressed_size']<cap and meta['compressed_size']<cap
 assert not meta['name'].startswith('/')and'..'not in Path(meta['name']).parts
 stem=hashlib.sha256(meta['name'].encode()).hexdigest()[:16];off=meta['offset'];head=E/'cache'/f'member_{stem}_header.bin'
 F.bounded_range(F.URL,off,off+29,head);x=struct.unpack('<4s5H3I2H',head.read_bytes());assert x[0]==b'PK\x03\x04'and x[3]==meta['method'];nl,el=x[-2:];assert nl+el<65536
 n=nl+el+meta['compressed_size'];body=E/'cache'/f'member_{stem}_data.bin';F.bounded_range(F.URL,off+30,off+29+n,body)
 raw=body.read_bytes();assert raw[:nl].decode('utf-8'if meta['flags']&0x800 else'cp437')==meta['name'];compressed=raw[nl+el:]
 if meta['method']==0:data=compressed
 elif meta['method']==8:
  obj=zlib.decompressobj(-15);data=obj.decompress(compressed,meta['uncompressed_size']+1);assert obj.eof and not obj.unconsumed_tail and not obj.unused_data
 else:raise ValueError('Unsupported ZIP method')
 assert len(data)==meta['uncompressed_size']and zlib.crc32(data)&0xffffffff==meta['crc32'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
 write(out.with_suffix(out.suffix+'.member.json'),{**meta,'url':F.URL,'etag':F.ETAG,'sha256':F.sha(out),'crc_verified':True,'body_range':[off+30,off+29+n]})
 return out
def metadata():
 initialize();allmeta=read(OLD/'results/central_summary.json')['non_frame_members']
 selected=[m for m in allmeta if m['name']=='task_ABC_D/training/lang_annotations/auto_lang_ann.npy'];assert len(selected)==1
 write(E/'configs/metadata_member.json',selected[0]);p=extract(selected[0],E/'data/metadata/auto_lang_ann.npy',metadata=True)
 with p.open('rb')as f:
  version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version);body=f.read()
 assert dtype.hasobject and shape==()
 counts=collections.Counter();globals_=[];strings=[];large=[]
 for op,arg,pos in pickletools.genops(body):
  counts[op.name]+=1
  if op.name=='GLOBAL':globals_.append(arg)
  if isinstance(arg,str)and len(strings)<140:strings.append({'opcode':op.name,'position':pos,'value':arg[:160],'length':len(arg)})
  if isinstance(arg,bytes)and len(arg)>1024:large.append({'position':pos,'length':len(arg),'sha256':hashlib.sha256(arg).hexdigest()})
 write(E/'results/metadata_opcode_inspection.json',{'path':str(p),'shape':list(shape),'dtype':str(dtype),'object_pickle_executed':False,'opcode_counts':dict(counts),'globals_as_text_only':globals_,'first_strings':strings,'large_byte_arrays':large,'cumulative_body_bytes':read(E/'results/network_ledger.json')['body_bytes_read']})
 print('METADATA_INSPECTED',dict(counts),'GLOBALS',globals_,'CUMULATIVE_BODY',read(E/'results/network_ledger.json')['body_bytes_read'],flush=True)
if __name__=='__main__':metadata()
