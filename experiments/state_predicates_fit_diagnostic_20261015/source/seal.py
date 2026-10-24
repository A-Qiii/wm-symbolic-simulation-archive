from common_diag import *
import re
q=frozen();assert not(E/'delivery_manifest.json').exists()
assert read(E/'results/final_training_audit.json')['passed']
assert read(E/'results/training_complete.json')['total_updates']==1500
assert read(E/'results/figure_manifest.json')['all4_visually_inspected']
assert read(E/'results/decision.json')['candidate_promoted']==False
assert read(E/'NEXT_STEP_PROPOSAL.json')['status']=='proposal_only_not_executed'
am=read(E/'configs/launch_fix.json');assert sha(am['corrected_source'])==am['corrected_source_sha256']and am['optimizer_updates_before_fix']==0
for doc in ['REPORT.md','HANDOFF.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(E/doc).read_text()):
  if target=='results/immutable_after.json':continue
  assert(E/target).exists(),(doc,target)
before=read(E/'results/immutable_before.json')['manifests'];after=verify_prior();assert before==after and len(after)==9
save(E/'results/immutable_after.json',{'created_utc':now(),'matches_before':True,'manifests':after})
save(E/'results/delivery_validation.json',{'created_utc':now(),'passed':True,'frozen_protocol_and_sources_unchanged':True,'document_links_valid':True,'four_figures_inspected':True,'old9_sealed_roots_unchanged':True,'C_D_forwards':0,'new_simulation':0,'updates':1500,'candidate_promoted':False,'original_five_pair_negative_report_retained':True,'syntax_fix_before_training_recorded':True})
entries=[]
for p in sorted(E.rglob('*')):
 if p.is_file():
  assert not p.is_symlink();entries.append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)})
save(E/'delivery_manifest.json',{'created_utc':now(),'root':str(E),'scope':'Bounded single-seed training-failure diagnosis; all temporary models diagnostic only','entries':entries,'total_bytes':sum(r['bytes']for r in entries),'protocol_sha256':sha(E/'configs/protocol.json'),'excludes':'delivery_manifest.json itself; manifest SHA in external receipt'})
for p in E.rglob('*'):
 if p.is_file():p.chmod(p.stat().st_mode&~0o222)
for r in entries:
 p=E/r['path'];assert sha(p)==r['sha256']and not(p.stat().st_mode&0o222)
files=[p for p in E.rglob('*')if p.is_file()];assert len(files)==len(entries)+1 and all(not(p.stat().st_mode&0o222)for p in files)
receipt={'created_utc':now(),'experiment':str(E),'report':str(E/'REPORT.md'),'handoff':str(E/'HANDOFF.md'),'outcome':'completed_bounded_diagnosis','manifest_sha256':sha(E/'delivery_manifest.json'),'entries_verified':len(entries),'files_read_only':len(files),'bytes_excluding_manifest':sum(r['bytes']for r in entries),'old9_sealed_roots_unchanged':True,'updates':1500,'seed':0,'C_D_forwards':0,'new_collection':0,'candidate_promoted':False,'final_results':{'five_only':{'five':5,'old132':72},'joint_original':{'five':5,'old132':132},'joint_equal_groups':{'five':5,'old132':132}},'suggested_single_change':'Fixed equal group weighting, not yet a candidate or generalization claim','next_step_executed':False}
save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/fit_diagnostic_delivery_receipt.json'),receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
