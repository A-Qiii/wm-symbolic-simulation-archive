from common import *
import shutil,re,collections

def main():
 manifest=E/'configs/delivery_manifest.json';assert not manifest.exists()
 for name in['pre_adjudication_frozen','labels_frozen','sensitivity_frozen','source_references']:check_manifest(E/f'configs/{name}.json')
 v=read(E/'results/verification.json');assert v['passed']
 for name,x in v['previous_seals_unchanged'].items():
  p=P/f'experiments/{name}/configs/delivery_manifest.json';assert sha(p)==x['sha256'];check_manifest(p)
 ann=read(E/'data/labels/adjudicated_eval_v1.json')['annotations'];assert sorted(a['image_id']for a in ann)==IDS
 assert sum(c['status']!='uncertain'for a in ann for c in a['classes'])==12 and sum(len(c['boxes'])for a in ann for c in a['classes'])==18
 counts=collections.Counter(x['action']for x in read(E/'data/labels/decision_ledger.json')['decisions']);assert counts['correct_visible_extent']==7 and counts['include_identifiable_visible_parts']==3 and counts['retain_original']==2
 assert len(read(E/'data/labels/external_arbitration_needed.json'))==4
 links=0
 for p in R.glob('*.md'):
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://'in link or link.startswith('#'):continue
   q=(p.parent/link.split('#')[0]).resolve();assert q==manifest or q.is_file(),(p,link);links+=1
 for path in Path('/tmp').glob('label8*20261018.log'):shutil.copy2(path,E/'logs'/path.name)
 assert not list(E.rglob('*.pt'))and not(E/'data/features').exists()and not(E/'data/train').exists()
 for name in['adjudicated_labels8','arbitration_cases','label_sensitivity8']:assert(R/f'figures/{name}.png').stat().st_size>10000
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 save(E/'results/completion_audit.json',{'utc':now(),'status':'completed_bounded_8image_label_adjudication_and_saved_prediction_sensitivity','selected_ids':IDS,'source_states':4,'selected_AB_development_images':0,'train_images':4,'C_seen_images':2,'D_seen_images':2,'old_known_class_images':11,'new_known_class_images':12,'old_reference_boxes':18,'new_reference_boxes':18,'common_known_class_images':9,'common_reference_boxes':14,'coordinate_corrections_class_images':7,'new_known_class_images_added':3,'old_known_class_images_dropped':2,'retained_original_class_images':2,'remaining_unknown_class_images':4,'external_arbitration_needed':['id0 fingers','id8 fingers','id1 drawer_front','id9 drawer_front'],'assistant_labels_not_human_gold':True,'not_blind_adjudicator_historical_context_exposure_disclosed':True,'labels_frozen_before_saved_output_read':True,'new_model_forwards':0,'new_training_steps':0,'new_feature_extractions':0,'new_decoding_runs':0,'new_images_or_downloads':0,'GPU_used':False,'all_selected_original_scores_reproduced':True,'no_old_labels_or_reports_modified':True,'previous_unchanged_seals':v['previous_seals_unchanged'],'full_data_labels_audit_incomplete':True,'other32_images_not_adjudicated':True,'no_full_data_score_recomputation_or_extrapolation':True,'no_AB_development_score_reestimate':True,'all_full_data_scores_provisional':True,'figures_visually_checked':['adjudicated_labels8.png','arbitration_cases.png','label_sensitivity8.png'],'presentation_only_repair':'Increased arbitration plot row spacing; label/score freeze hashes unchanged','links_checked':links,'syntax_failure_preserved':'prepare first invocation rejected else10 before any task code execution; fixed then successful','no_resource_or_permission_blockers':True,'remaining_semantic_blocker':'External component identity/extent adjudication for4unknown cells; audit of remaining32images incomplete','new_experiments_stopped':True,'overall_guide_complete':False})
 save(E/'logs/seal_checks.json',{'utc':now(),'passed':True,'links_checked':links,'old_files_unchanged':sum(x['files_verified']for x in v['previous_seals_unchanged'].values()),'labels_and_scores_unchanged_since_freeze':True,'all_new_figures_inspected':True})
 files=sorted(p for root in[E,R]for p in root.rglob('*')if p.is_file()and p!=manifest);sizes={str(p):p.stat().st_size for p in files}
 save(manifest,{'utc':now(),'status':'sealed','files':len(files),'bytes':sum(sizes.values()),'sha256':{str(p):sha(p)for p in files},'sizes':sizes,'manifest_excludes_itself':True,'scope':'Exactly8RGB label adjudication and saved-prediction sensitivity only; full-data label audit unfinished, no extrapolation or new model work'})
 check_manifest(manifest);print('EIGHT_IMAGE_LABEL_AUDIT_SEALED',len(files),sum(sizes.values()),sha(manifest),flush=True)
if __name__=='__main__':main()
