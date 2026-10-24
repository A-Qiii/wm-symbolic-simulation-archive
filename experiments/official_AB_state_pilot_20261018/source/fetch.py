"""Bounded read-only extraction of named official ZIP members. Never unpickle."""
from pathlib import Path
import json,struct,urllib.request,hashlib,time,zlib,datetime,sys,shutil,pickletools,io
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_AB_state_pilot_20261018'
URL='http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip';SIZE=555309812705;ETAG='"814b0b4be1-5e8bad7e824c0"';CDSTART=555080601096;CDSIZE=229211511;BUDGET=512*1024**2
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
def initialize():
 for sub in ['source','configs','cache','data/metadata','data/frames','results','logs']:(E/sub).mkdir(parents=True,exist_ok=True)
 p=E/'configs/protocol.json'
 if not p.exists():write(p,{'utc':now(),'authorization':'Parent human-approved official-data pilot, total<=512MiB including central directory; no training','url':URL,'source_size':SIZE,'source_etag':ETAG,'central_directory_offset':CDSTART,'central_directory_bytes':CDSIZE,'total_body_read_cap':BUDGET,'untrusted_pickle':'disassemble only, never pickle.load or np.load allow_pickle=True','data_scope':'A/B episodes only, selected after official scene mapping and episode boundary verification; human play only state labels, no fixed-controller transitions','new_model_updates':0})
 shutil.copy2(__file__,E/'source/fetch.py')
 if not(E/'results/network_ledger.json').exists():write(E/'results/network_ledger.json',{'body_bytes_read':0,'requests':[]})
def bounded_range(url,start,end,path,total=SIZE,etag=ETAG):
 path=Path(path);receipt=path.with_suffix(path.suffix+'.receipt.json')
 if path.exists()and receipt.exists():
  r=read(receipt);assert r['sha256']==sha(path)and r['start']==start and r['end']==end and r['url']==url;return path
 n=end-start+1;ledger=read(E/'results/network_ledger.json');assert n>0 and ledger['body_bytes_read']+n<=BUDGET
 rec={'utc':now(),'url':url,'start':start,'end':end,'expected_bytes':n,'body_bytes_read':0,'status':'requesting'};ledger['requests'].append(rec);write(E/'results/network_ledger.json',ledger)
 req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'CALVIN-local-state-label-pilot/1.0'})
 part=path.with_suffix(path.suffix+'.partial');path.parent.mkdir(parents=True,exist_ok=True)
 try:
  with urllib.request.urlopen(req,timeout=30)as response:
   rec.update({'http_status':response.status,'content_range':response.headers.get('Content-Range'),'content_length':response.headers.get('Content-Length'),'etag':response.headers.get('ETag'),'resolved_url':response.url})
   # Reject ignored ranges before reading any response body.
   assert response.status==206,rec
   assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}',rec
   assert int(response.headers.get('Content-Length','-1'))==n,rec
   if etag is not None:assert response.headers.get('ETag')==etag,rec
   assert response.url==url,rec
   with part.open('wb')as f:
    while rec['body_bytes_read']<n:
     b=response.read(min(1024*1024,n-rec['body_bytes_read']));assert b,'truncated response';f.write(b);rec['body_bytes_read']+=len(b);ledger['body_bytes_read']+=len(b);write(E/'results/network_ledger.json',ledger)
   assert rec['body_bytes_read']==n
  part.rename(path);rec.update({'status':'complete','sha256':sha(path)});write(receipt,rec);write(E/'results/network_ledger.json',ledger);return path
 except BaseException as ex:
  rec.update({'status':'failed','error':repr(ex)});write(E/'results/network_ledger.json',ledger);raise
def directory():
 dst=E/'cache/ABC_central_directory.bin'
 if dst.exists():assert dst.stat().st_size==CDSIZE;return dst
 chunks=[]
 for offset in range(0,CDSIZE,8*1024**2):
  n=min(8*1024**2,CDSIZE-offset);p=E/'cache'/f'central_{offset:09d}.bin';bounded_range(URL,CDSTART+offset,CDSTART+offset+n-1,p);chunks.append(p);print('CENTRAL_BYTES',offset+n,'/',CDSIZE,flush=True)
 with dst.open('wb')as f:
  for p in chunks:
   with p.open('rb')as x:shutil.copyfileobj(x,f)
 assert dst.stat().st_size==CDSIZE;write(E/'results/central_directory.json',{'bytes':CDSIZE,'sha256':sha(dst),'chunks':[str(x)for x in chunks],'utc':now()});return dst
def entries():
 with (E/'cache/ABC_central_directory.bin').open('rb')as f:
  while True:
   b=f.read(46)
   if not b:break
   assert len(b)==46 and b[:4]==b'PK\x01\x02'
   x=struct.unpack('<4s6H3I5H2I',b);_,made,needed,flags,method,mt,md,crc,cs,us,nl,el,cl,disk,ia,ea,off=x
   name=f.read(nl).decode('utf-8'if flags&0x800 else'cp437');extra=f.read(el);f.read(cl)
   assert not flags&1,'encrypted archive unsupported'
   pos=0
   while pos+4<=len(extra):
    kind,sz=struct.unpack_from('<HH',extra,pos);v=extra[pos+4:pos+4+sz];pos+=4+sz
    if kind==1:
     j=0
     if us==0xffffffff:us=struct.unpack_from('<Q',v,j)[0];j+=8
     if cs==0xffffffff:cs=struct.unpack_from('<Q',v,j)[0];j+=8
     if off==0xffffffff:off=struct.unpack_from('<Q',v,j)[0];j+=8
   yield {'name':name,'compressed_size':cs,'uncompressed_size':us,'offset':off,'crc32':crc,'method':method,'flags':flags}
def extract_member(meta,out):
 out=Path(out);receipt=out.with_suffix(out.suffix+'.member.json')
 if out.exists()and receipt.exists():assert sha(out)==read(receipt)['sha256'];return out
 assert meta['uncompressed_size']<16*1024**2 and meta['compressed_size']<16*1024**2,meta
 assert not meta['name'].startswith('/')and'..'not in Path(meta['name']).parts
 stem=hashlib.sha256(meta['name'].encode()).hexdigest()[:16];off=meta['offset'];head=E/'cache'/f'member_{stem}_header.bin';bounded_range(URL,off,off+29,head);b=head.read_bytes();x=struct.unpack('<4s5H3I2H',b);assert x[0]==b'PK\x03\x04';assert x[3]==meta['method'];nl,el=x[-2:];assert nl+el<65536
 payload=E/'cache'/f'member_{stem}_data.bin';n=nl+el+meta['compressed_size'];bounded_range(URL,off+30,off+29+n,payload);raw=payload.read_bytes();name=raw[:nl].decode('utf-8'if meta['flags']&0x800 else'cp437');assert name==meta['name'];compressed=raw[nl+el:]
 if meta['method']==0:data=compressed
 elif meta['method']==8:
  obj=zlib.decompressobj(-15);data=obj.decompress(compressed,meta['uncompressed_size']+1);assert obj.eof and not obj.unconsumed_tail and not obj.unused_data
 else:raise ValueError(('unsupported ZIP method',meta['method']))
 assert len(data)==meta['uncompressed_size'];assert zlib.crc32(data)&0xffffffff==meta['crc32'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);write(receipt,{'utc':now(),**meta,'url':URL,'etag':ETAG,'sha256':sha(out),'crc_verified':True,'local_header_range':[off,off+29],'member_range':[off+30,off+29+n]});return out
def disassemble_npy(path):
 with Path(path).open('rb')as f:
  version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version);body=f.read()
 if dtype.hasobject:
  buff=io.StringIO();pickletools.dis(body,out=buff);dest=Path(path).with_suffix('.pickle_disassembly.txt');dest.write_text(buff.getvalue());return {'path':str(path),'shape':list(shape),'dtype':str(dtype),'object_pickle_executed':False,'disassembly':str(dest)}
 x=np.load(path,allow_pickle=False);return {'path':str(path),'shape':list(shape),'dtype':str(dtype),'data':x.tolist()}
def metadata():
 directory();selected=[];counts={'total_members':0,'training_frame_members':0,'validation_frame_members':0};names=[]
 for e in entries():
  counts['total_members']+=1;n=e['name']
  if '/training/episode_'in n and n.endswith('.npz'):counts['training_frame_members']+=1
  if '/validation/episode_'in n and n.endswith('.npz'):counts['validation_frame_members']+=1
  if not n.endswith('/')and'/episode_'not in n:names.append(e)
  if '/training/'in n and(Path(n).name in ['ep_start_end_ids.npy','scene_info.npy','statistics.yaml']or'/.hydra/'in n and Path(n).name in ['config.yaml','merged_config.yaml','overrides.yaml']):selected.append(e)
 assert counts['total_members']==1894126,counts;write(E/'results/central_summary.json',counts|{'non_frame_members':names,'selected_metadata':selected});print('CENTRAL_PARSED',counts,flush=True)
 for e in selected:
  rel=e['name'].split('/training/',1)[1];p=extract_member(e,E/'data/metadata/training'/rel);print('METADATA',rel,e['compressed_size'],flush=True)
  if p.suffix=='.npy':write(p.with_suffix('.inspection.json'),disassemble_npy(p))
 fix='http://calvin.cs.uni-freiburg.de/scene_info_fix/task_ABC_D_scene_info.zip';p=bounded_range(fix,0,524,E/'cache/task_ABC_D_scene_info.zip',total=525,etag=None)
 import zipfile
 with zipfile.ZipFile(p)as z:
  for info in z.infolist():
   if info.is_dir():continue
   assert not info.filename.startswith('/')and'..'not in Path(info.filename).parts;assert info.file_size<65536
   data=z.read(info);assert zlib.crc32(data)&0xffffffff==info.CRC;out=E/'data/metadata/scene_info_fix'/info.filename;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);write(out.with_suffix('.inspection.json'),disassemble_npy(out))
 print('METADATA_COMPLETE_BYTES',read(E/'results/network_ledger.json')['body_bytes_read'],flush=True)
if __name__=='__main__':
 initialize();metadata()
