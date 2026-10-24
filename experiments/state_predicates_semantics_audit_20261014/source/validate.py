from pathlib import Path
import json,hashlib,datetime
from PIL import Image
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_semantics_audit_20261014';A=P/'experiments/state_predicates_interface_acceptance_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not(E/'results/validation.json').exists();j=read(E/'results/audit.json');w=read(E/'results/window_inventory.json');r=read(E/'results/raw_place_coverage.json');checks=[]
 assert len(w)==1364 and sum(x['split']=='train'for x in w)==1009 and sum(x['split']=='dev'for x in w)==355
 assert all(x['environment']in 'AB'if x['split']=='train'else x['environment']=='C'for x in w)
 assert j['input_audit']['identity_compared']==4092 and not j['input_audit']['identity_mismatch'];assert max(j['input_audit']['max_tensor_error'].values())==0
 for x in w:
  if x['source']=='long':
   ids=[int(Path(p).stem)for p in x['actual_public_paths']];assert ids==[ids[-1]-d for d in [63,42,21,0]]
 assert len(r['trajectories'])==296 and sum(x['frames_scanned']for x in r['summary'])==80340
 seed=[q for q in j['C_saved_prediction_scores']if q['seed']==1];assert sum(x['posthoc_full_physical_valid']['valid_facts']for x in seed)==2485;assert sum(x['posthoc_full_physical_valid']['raw_FP']+x['posthoc_full_physical_valid']['raw_FN']+x['posthoc_full_physical_valid']['UNKNOWN']for x in seed)==0
 # Recheck every byte covered by the completed acceptance delivery manifests.
 for mp in [A/'delivery_manifest.json',Path(read(A/'delivery_manifest.json')['dataset_manifest'])]:
  m=read(mp);root=Path(m['root']);total=0
  for q in m['entries']:
   p=root/q['path'];assert p.stat().st_size==q['bytes']and sha(p)==q['sha256'],str(p);total+=q['bytes']
  checks.append({'manifest':str(mp),'manifest_sha256':sha(mp),'files_verified':len(m['entries']),'bytes_verified':total})
 assert sha(Path(read(A/'delivery_manifest.json')['dataset_manifest']))==read(A/'delivery_manifest.json')['dataset_manifest_sha256']
 frozen=read(E/'configs/protocol.json')['frozen_sha256'];frozen.update(read(A/'configs/protocol.json')['frozen_sha256'])
 for p,h in frozen.items():assert sha(p)==h,p
 for p in [E/'figures/physical_semantics.png',E/'figures/coverage_masks.png']:
  with Image.open(p)as im:im.verify()
 result={'passed':True,'scope':'Finite read-only semantics/coverage audit and proposal complete; no implementation/training/collection/WM run.','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'train_windows':1009,'C_windows':355,'mapped_entities_checked':4092,'cached_tensor_reconstruction_max_absolute_error':0,'raw_place_trajectories':296,'raw_place_frames':80340,'seed1_C_all_valid_facts':2485,'seed1_C_raw_errors':0,'new_inference_batches':0,'training_updates':0,'threshold_or_decoder_changes':0,'original_acceptance_complete_manifest_verification':checks,'frozen_code_weights_protocol_files_unchanged':len(frozen),'images_visually_reviewed':True,'remaining_limits':['C predictions are existing saved outputs; full labels were evaluated posthoc.','No saved training predictions: cannot assert training errors absent.','Only actual-model windows and saved core place traces plus existing6Cdiagnostic geometry were audited; not exhaustive all raw traces.','Future long-history macro-transition inputs remain incomplete.','No causal intervention established why the two acceptance reader errors occurred.','Full original guide remains incomplete.']}
 (E/'results/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 manifest={'root':str(E),'algorithm':'SHA256','entries':[]}
 for p in sorted(E.rglob('*')):
  if p.is_file()and p.name!='delivery_manifest.json':manifest['entries'].append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)})
 manifest['files']=len(manifest['entries']);manifest['bytes']=sum(x['bytes']for x in manifest['entries']);(E/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
