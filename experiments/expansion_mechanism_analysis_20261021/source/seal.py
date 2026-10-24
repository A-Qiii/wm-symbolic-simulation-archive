from pathlib import Path
import json,hashlib,datetime,stat,shutil
B=Path('/home/robot/research/projects/wm_symbolic/experiments/expansion_mechanism_analysis_20261021');W=Path('/home/robot/Documents/Codex/2026-10-13/task-4');v=json.loads((B/'results/final_verification.json').read_text());assert v['passed'];assert not(B/'MANIFEST.json').exists();shutil.copy2(W/'expansion_mechanism_seal.py',B/'source/seal.py')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
files=[]
for p in sorted(B.rglob('*')):
 assert not p.is_symlink(),p
 if p.is_file():files.append({'path':str(p.relative_to(B)),'bytes':p.stat().st_size,'sha256':sha(p)})
m={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(B),'scope':'Completed existing-data matched-window, paired expansion, control-mechanism, architecture and objective/optimization audit. No new model inference, training, data collection, tuning or original edits. Excludes manifest and checksum.','files':files,'file_count':len(files),'total_bytes':sum(r['bytes']for r in files),'readonly_after_seal':True};(B/'MANIFEST.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');h=sha(B/'MANIFEST.json');(B/'MANIFEST.sha256').write_text(h+'  MANIFEST.json\n')
for p in sorted(B.rglob('*'),reverse=True):p.chmod(stat.S_IMODE(p.stat().st_mode)&~0o222)
B.chmod(stat.S_IMODE(B.stat().st_mode)&~0o222)
for r in files:
 p=B/r['path'];assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'];assert not p.stat().st_mode&0o222
assert sha(B/'MANIFEST.json')==h
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'snapshot_files_verified':len(files),'snapshot_bytes':m['total_bytes'],'manifest_sha256':h,'readonly':True,'original_protected_files_unchanged':sum(v['original_snapshots_all_files_unchanged_readonly'].values()),'additional_raw_hashes_unchanged':v['additional_raw_file_hashes_unchanged'],'matched_windows_independently_checked':v['matched_windows_recomputed'],'no_training_or_model_inference_or_collection':True};(W/'expansion_mechanism_seal_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
