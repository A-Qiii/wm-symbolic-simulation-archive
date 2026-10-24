"""CPU-only preparation; validates compatibility and freezes first20 of official1000."""
from pathlib import Path
import sys,re,ast,json,tarfile,hashlib,datetime,itertools
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/official_standard_subset_20261014'
out.mkdir(exist_ok=False)
sys.path.insert(0,str(P/'scripts/official_compat'))
import pyhash
archive=P/'cache/dependency_sources/pyhash-0.9.3.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='cff5c81d613163fc59d623d4546d9be55b46ecd0e573b59057b1bb112a497763'
with tarfile.open(archive) as t:
 source=t.extractfile('pyhash-0.9.3/src/fnv/test_fnv.c').read().decode()
 src=t.extractfile('pyhash-0.9.3/src/Hash.h').read().decode()
 assert 'PyUnicode_AsUTF16String' in src and 'py::arg("seed") = 0' in src
 data=source.split('struct test_vector fnv_test_str[] = {',1)[1].split('{NULL, 0}',1)[0]
 entries=re.findall(r'\b(TEST0?)\((?:(R10|R500)\()?("(?:\\.|[^"\\])*")\)?\)',data)
 vectors=[]
 for kind,repeat,literal in entries:
  value=ast.literal_eval('b'+literal)*(int(repeat[1:]) if repeat else 1)
  vectors.append(value+(b'\0' if kind=='TEST0' else b''))
 checks=0
 for name,seed in [('fnv0_32',0),('fnv1_32',0x811c9dc5)]:
  section=source.split(f'{name}_vector[] = {{',1)[1].split('};',1)[0]
  expected=re.findall(r'fnv_test_str\[(\d+)\].*?0x([0-9a-fA-F]+)UL',section)
  assert len(expected)==len(vectors),(len(expected),len(vectors))
  for index,value in expected:
   assert pyhash.fnv1_32(seed)(vectors[int(index)])==int(value,16),(name,index)
   checks+=1
 assert pyhash.fnv1_32()('a')==0x610098b3 # authoritative vector13 = b'a\0'
 assert pyhash.fnv1_32()('a')!=pyhash.fnv1_32()(b'a')
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
import numpy as np
all_sequences=get_sequences(1000,num_workers=1)
assert len(all_sequences)==1000
unique={str(initial.values()):initial for initial,_ in all_sequences}
for text in unique:
 # Independently transcribed GCC optimized shift/add expression in official C.
 h=0
 for byte in text.encode('utf-16-le'):
  h=(h+(h<<1)+(h<<4)+(h<<7)+(h<<8)+(h<<24))&0xffffffff;h^=byte
 assert h==pyhash.fnv1_32()(text)
all_serial=[{'index':i,'initial_condition':initial,'skills':list(sequence)} for i,(initial,sequence) in enumerate(all_sequences)]
all_bytes=json.dumps(all_serial,ensure_ascii=False,separators=(',',':')).encode()
(out/'official_1000_sequence_manifest.json').write_bytes(all_bytes)
selected=[]
for item in all_serial[:20]:
 robot,scene=get_env_state_for_initial_condition(item['initial_condition']);folder=out/f"sequence_{item['index']:04d}";folder.mkdir()
 np.savez_compressed(folder/'intended_official_reset.npz',robot_obs=robot,scene_obs=scene)
 selected.append({**item,'initial_condition_hash_seed':pyhash.fnv1_32()(str(item['initial_condition'].values())), 'reset_npz':str(folder/'intended_official_reset.npz')})
record={'status':'prepared_not_run','prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'subset_rule':'first20 indices0..19 of get_sequences(1000,num_workers=1), no success filtering','official_full_sequence_count':1000,'diagnostic_sequence_count':20,'official_sequence_manifest_sha256':hashlib.sha256(all_bytes).hexdigest(),'policy_random_seed':0,'world_model_or_main_D_tuning':False,'max_controls_per_subtask':360,'subtask_stopping':'official task oracle success, isolated official diagnosis only','visual_stopper_enabled':False,'pyhash_compatibility':{'native_package_installed':False,'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'source_test_vectors_passed':checks,'unicode_byte_semantics_test_passed':True,'initial_condition_shift_add_crosschecks':len(unique),'implementation':str(P/'scripts/official_compat/pyhash.py'),'implementation_sha256':hashlib.sha256((P/'scripts/official_compat/pyhash.py').read_bytes()).hexdigest(),'limitation':'Compatibility implementation, not compiled pyhash package; no claim of full historical runtime equivalence'},'selected':selected}
(out/'protocol.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print('PREPARED',len(selected),'TESTS',checks,'STATES',len(unique),'MANIFEST',record['official_sequence_manifest_sha256'],flush=True)
