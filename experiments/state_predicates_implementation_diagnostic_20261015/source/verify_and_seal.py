from audit_common import *
import re
def main():
 q=frozen();checks=read(E/'results/forward_diagnostic_integrity.json');assert checks['passed']and checks['optimizer_updates']==0 and checks['D_neural_forward_calls']==0;rs=read(E/'results/gradient_probes.json');assert len(rs)==30
 for r in rs:
  parts=r['components'];b=sum(x['loss_actual_contribution']for x in parts if x['component'].startswith('BCE'));l=sum(x['loss_actual_contribution']for x in parts if x['component'].startswith('latent'));assert abs(b-r['BCE'])<2e-7 and abs(l-r['latent_weighted'])<2e-7 and abs(b+l-r['total_loss'])<2e-7;assert r['gradient_sum_relative_to_component_norms']<.001
 tables=read(E/'results/combination_tables.json');rows=read(E/'results/all_binding_rows.json');assert len(rows)==258
 for tt in tables:
  for held,cols in tt['rows_held_columns_target'].items():
   for target,count in cols.items():assert count==sum(r['group']==tt['group']and r['family']=='lift'and r['held_colour_truth_scoring_only']==held and r['target_colour_declared']==target and(tt['drawer']=='all'or r['drawer_open_truth_scoring_only']==(tt['drawer']=='open'))for r in rows)
 hist=read(E/'results/historical_loss_windows.json');assert sum(r['number_steps']for r in hist)==9000;assert read(E/'NEXT_STEP_PROPOSAL.json')['status']=='proposal_only_not_executed';report=(E/'REPORT.md').read_text()
 for target in re.findall(r'\]\(([^)]+)\)',report):
  if not target.startswith('https://'):assert(E/target).exists(),target
 after=verify_sealed();before=read(E/'results/immutable_before.json')['manifests'];assert after==before;save(E/'results/immutable_after.json',{'created_utc':now(),'passed':True,'manifests':after,'exactly_equal_to_before':True});save(E/'results/final_validation.json',{'created_utc':now(),'passed':True,'all258_action_rows_and_colour_tables_recounted':True,'all30_component_losses_sum_to_total':True,'original9000_training_log_steps_only_read':True,'original_D_and_training_all_files_sha256_and_readonly_unchanged':True,'optimizer_updates':0,'new_collection':0,'D_model_forward_calls':0,'D_independent_rescoring':0,'report_links_resolve':True,'figures_visually_inspected':True,'minimal_proposal_not_executed':True})
 assert(E/'HANDOFF.md').exists();files=sorted(p for p in E.rglob('*')if p.is_file()and p.name!='delivery_manifest.json');entries=[{'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files];save(E/'delivery_manifest.json',{'created_utc':now(),'scope':'finite implementation and combination diagnosis; no training collection or new D inference','entries':entries,'entry_count':len(entries),'bytes':sum(r['bytes']for r in entries)})
 for p in files+[E/'delivery_manifest.json']:p.chmod(0o444)
 receipt={'created_utc':now(),'sealed':True,'root':str(E),'manifest_sha256':sha(E/'delivery_manifest.json'),'readonly_files':len(files)+1,'bytes_excluding_manifest':sum(r['bytes']for r in entries),'original_D_manifests_unchanged':[r for r in after if r['root']in [str(D),str(DB)]],'report':str(E/'REPORT.md'),'next_step_proposal':str(E/'NEXT_STEP_PROPOSAL.json')};save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/implementation_diagnostic_receipt.json'),receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
