from common_control import *
import re
q=frozen();assert not(E/'delivery_manifest.json').exists();ta=read(E/'results/training_audit.json');ea=read(E/'results/evaluation_audit.json');dec=read(E/'results/decision.json');fig=read(E/'results/figure_manifest.json')
assert ta['passed']and ea['passed']and ta['total_new_optimizer_updates']==3000;assert dec['stable_weight_advantage']and not any(dec['complete_restoration_all3seeds'].values());assert not dec['candidate_promoted'];assert len(fig['files'])==5 and fig['visually_inspected']
for p in fig['files']:assert Path(p).exists()
for doc in ['REPORT.md','HISTORY_COMPARISON.md','HANDOFF.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(E/doc).read_text()):
  if target=='results/immutable_after.json':continue
  assert(E/target).exists(),(doc,target)
before=read(E/'results/immutable_before.json')['manifests'];after=verify_prior();assert before==after and len(after)==10
save(E/'results/immutable_after.json',{'created_utc':now(),'matches_before':True,'manifests':after})
save(E/'results/delivery_validation.json',{'created_utc':now(),'passed':True,'frozen_core_sources_and_protocol_unchanged':True,'reports_links_valid':True,'five_figures_visually_inspected':True,'old10_sealed_roots_unchanged':True,'actual_new_updates':3000,'C_evaluated_only_after_training':True,'D_forward_calls':0,'new_simulation':0,'additional_adaptive_updates':0,'candidate_promoted':False,'original_negative_reports_preserved':True})
entries=[]
for p in sorted(E.rglob('*')):
 if p.is_file():assert not p.is_symlink();entries.append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)})
save(E/'delivery_manifest.json',{'created_utc':now(),'root':str(E),'scope':'Three-seed paired group-loss-weight control; positive additional development effect with incomplete restoration and latent tradeoff','entries':entries,'total_bytes':sum(r['bytes']for r in entries),'protocol_sha256':sha(E/'configs/protocol.json'),'excludes':'delivery_manifest.json itself; manifest SHA stored in external receipt'})
for p in E.rglob('*'):
 if p.is_file():p.chmod(p.stat().st_mode&~0o222)
for r in entries:
 p=E/r['path'];assert sha(p)==r['sha256']and not(p.stat().st_mode&0o222)
files=[p for p in E.rglob('*')if p.is_file()];assert len(files)==len(entries)+1 and all(not(p.stat().st_mode&0o222)for p in files)
receipt={'created_utc':now(),'experiment':str(E),'report':str(E/'REPORT.md'),'handoff':str(E/'HANDOFF.md'),'outcome':'completed_control_positive_extra_symbolic_benefit_not_full_restoration','manifest_sha256':sha(E/'delivery_manifest.json'),'entries_verified':len(entries),'files_read_only':len(files),'bytes_excluding_manifest':sum(r['bytes']for r in entries),'old10_sealed_roots_unchanged':True,'actual_new_optimizer_updates':3000,'seeds':[0,1,2],'updates_each':500,'same_paired_start_optimizer_batches_budget':True,'sole_change':'loss group weights','drawer_C_all7':{'uniform':[10,9,5],'equal_groups':[11,11,12],'denominator':12},'drawer_both_sides':{'uniform':[4,4,0],'equal_groups':[5,5,6],'denominator':6},'legacy_C24_and_C18_all_correct':True,'stable_weight_advantage':True,'complete_restoration_all3seeds':False,'D_forward_calls':0,'new_collection':0,'candidate_promoted':False,'independent_validation_not_started':True}
save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/group_weight_delivery_receipt.json'),receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
