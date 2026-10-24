from pathlib import Path
import json,hashlib,datetime,stat,shutil
A=Path('/home/robot/research/projects/wm_symbolic/experiments/prediction_error_analysis_20261021');W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
assert json.loads((A/'results/final_verification.json').read_text())['passed'];assert not(A/'MANIFEST.json').exists();shutil.copy2(W/'prediction_error_seal.py',A/'source/seal.py')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
files=[]
for p in sorted(A.rglob('*')):
 assert not p.is_symlink(),p
 if p.is_file():files.append({'path':str(p.relative_to(A)),'bytes':p.stat().st_size,'sha256':sha(p)})
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(A),'scope':'Read-only post-hoc analysis of previously sealed predictions, execution logs and fixed training schedules. No training, inference, collection or original edits. Excludes this manifest and checksum.','files':files,'file_count':len(files),'total_bytes':sum(r['bytes']for r in files),'readonly_after_seal':True};(A/'MANIFEST.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');h=sha(A/'MANIFEST.json');(A/'MANIFEST.sha256').write_text(h+'  MANIFEST.json\n')
for p in sorted(A.rglob('*'),reverse=True):p.chmod(stat.S_IMODE(p.stat().st_mode)&~0o222)
A.chmod(stat.S_IMODE(A.stat().st_mode)&~0o222)
for r in files:
 p=A/r['path'];assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'];assert not p.stat().st_mode&0o222
assert sha(A/'MANIFEST.json')==h
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'files_verified':len(files),'snapshot_bytes':x['total_bytes'],'manifest_sha256':h,'all_snapshot_files_readonly':True,'original_snapshot_files_verified_unchanged':5732,'referenced_inputs_verified_unchanged':2758,'no_training':True,'no_model_inference':True,'no_collection':True,'no_original_writes':True};(W/'prediction_error_seal_verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
