from common import *
import shutil,re

def main():
 manifest=E/'configs/delivery_manifest.json';assert not manifest.exists(),'Already sealed'
 phases={}
 for name in['pre_decode_frozen','source_references','outputs_frozen_before_scoring','evaluation_frozen']:
  p=E/f'configs/{name}.json';phases[name]={'files':check_manifest(p),'sha256':sha(p)}
 v=read(E/'results/verification.json');assert v['passed']
 for name,entry in v['previous_seals_unchanged'].items():
  p=P/f'experiments/{name}/configs/delivery_manifest.json';assert sha(p)==entry['sha256'];check_manifest(p)
 d=read(E/'results/decoding.json');assert d['optimizer_steps']==d['feature_extractions']==d['new_forward_passes']==0
 assert d['raw_patch_class_candidates']==204800 and d['score_passing_candidates']==3457
 assert not list(E.rglob('*.pt'))and not(E/'data/features').exists()and not(E/'results/training_attempts.json').exists()
 for name in['prepare','decode','evaluate','verify','report']:
  p=Path(f'/tmp/endpoint_decoder_{name}_20261018.log');assert p.is_file();shutil.copy2(p,E/'logs'/p.name)
 shutil.copy2(__file__,E/'source/seal.py');links=0
 for p in[R/'REPORT.md',R/'STAGE_SUMMARY.md',R/'RUN_COMMANDS.md']:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://'in link or link.startswith('#'):continue
   q=(p.parent/link.split('#')[0]).resolve();assert q==manifest or q.is_file(),(p,link);links+=1
 save(E/'results/completion_audit.json',{'utc':now(),'status':'completed_bounded_saved_candidate_decoding_diagnostic',
  'source_RGB_images':40,'known_boxes':91,'model_configurations':4,'scored_variants':['original','corrected'],
  'corrected_rule':'No local-peak mutual exclusion or cardinality caps; unchanged score>=.5 and greedyclassNMSIoU>.3; every retained unmatched box isFP on knownclassimages',
  'prior_nopeak_diagnostic_incomplete_for_this_version':'Previous224diagnostic retained1/2caps; not reused as if equivalent',
  'new_training':0,'new_feature_extractions':0,'new_model_forwards':0,'new_data_or_model_downloads':0,'GPU_used':False,
  'label_read_during_decoding':0,'outputs_frozen_before_scoring':True,'GT_never_filters_predictions':True,
  'confirmed_design_issues':['same-class3x3peak suppression removes adjacent true-instance centers','earlycap lets incorrect high-score boxes consume true-instance slots'],
  'NMS_implementation_independently_reproduced':True,
  'train224_finger_TP_before_after_seeds':[[26,29],[27,30]],
  'ABdev_corrected_finger_TP_FP_FN':{'224seed0':[5,8,24],'224seed1':[2,7,27],'448seed0':[6,10,23],'448seed1':[5,1,24]},
  '448_AB_remaining_no_qualified_candidate_FN':[22,24],'448_AB_remaining_NMS_removed_FN':[1,0],
  'unresolved':'Generalization fails; attribution between coverage, representation, supervision/scoring and annotation accuracy remains unknown',
  'uncertain_classes_not_negative':True,'not_visible_examples':0,'same_assistant_repeat_not_independent_accuracy':True,'C_D_seen_diagnostic_only':True,
  'model_adopted_or_deployed':False,'relation_stage_started':False,'overall_guide_complete':False,
  'next_route_only':'Independently audited task-defined localization data with unseen roots and fixed standard detection/segmentation pipeline, then predicted-endpoint relationship/worldmodel study; not executed',
  'figure_inspected':'decoder_examples.png:3preselectedcases,all retainedfingerpredictions,blueGT/greenTP/redFP,paired identical RGB',
  'all_splits_resolutions_seeds_reported':True,'new_variants_count':1,'links_checked':links,'phase_checks':phases,'resource_or_permission_blockers':[]})
 save(E/'logs/seal_checks.json',{'utc':now(),'passed':True,'links_checked':links,'previous_seals_unchanged':list(v['previous_seals_unchanged']),'optimizer_model_feature_calls_zero':True})
 files=sorted(p for base in[E,R]for p in base.rglob('*')if p.is_file()and p!=manifest);hashes={str(p):sha(p)for p in files};sizes={str(p):p.stat().st_size for p in files}
 save(manifest,{'utc':now(),'status':'sealed','files':len(files),'bytes':sum(sizes.values()),'sha256':hashes,'sizes':sizes,'manifest_excludes_itself':True,'scope':'Saved-output uncapped/nopeak decoder diagnostic only; known design flaws identified, generalization fails; no adoption'})
 check_manifest(manifest);print('DECODER_DIAGNOSTIC_SEALED',len(files),sum(sizes.values()),sha(manifest),flush=True)
if __name__=='__main__':main()
