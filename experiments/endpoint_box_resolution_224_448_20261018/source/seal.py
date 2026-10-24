from common import *
import shutil,re,subprocess

def main():
 dest=E/'configs/delivery_manifest.json';assert not dest.exists(),'Already sealed'
 phases={}
 for name in ['pre_execution_frozen','source_references','training_frozen','final_heads_frozen','evaluation_frozen']:
  p=E/f'configs/{name}.json';phases[name]={'files':check_manifest(p),'sha256':sha(p)}
 verification=read(E/'results/verification.json');assert verification['passed']
 for name,entry in verification['previous_seals_unchanged'].items():
  p=P/f'experiments/{name}/configs/delivery_manifest.json';assert sha(p)==entry['manifest_sha256'];check_manifest(p)
 attempts=read(E/'results/training_attempts.json');assert len(attempts)==2 and all(x['resolution']==448 and x['status']=='completed'for x in attempts)
 assert len(list((E/'models/r448').glob('*update*.pt')))==2
 assert len(list((E/'models/r224').glob('*update*.pt')))==2
 for seed in(0,1):assert sha(E/f'models/r224/localizer_seed{seed}_update0600.pt')==sha(BASE/f'models/localizer_seed{seed}_update0600.pt')
 for r in read(E/'results/training.json')['runs']:assert r['updates']==600
 assert not list((P/'experiments/readout_supcon_ablation_20261018').rglob('training_attempts.json'))
 assert not list((P/'experiments/readout_supcon_ablation_20261018').rglob('*update0600.pt'))
 assert read(E/'results/preflight_and_reuse.json')['reuse224']
 assert read(E/'results/evaluation.json')['224_predictions_and_metrics_exactly_reproduced']
 for name in ['prepare','features','freeze','train','evaluate','verify','report','report_final']:
  p=Path(f'/tmp/endpoint_resolution_{name}_20261018.log');assert p.is_file();shutil.copy2(p,E/'logs'/p.name)
 shutil.copy2(__file__,E/'source/seal.py')
 links=0
 for p in[R/'REPORT.md',R/'RUN_COMMANDS.md']:
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://'in target or target.startswith('#'):continue
   f=(p.parent/target.split('#')[0]).resolve();assert f==dest or f.is_file(),(p,target);links+=1
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True);assert gpu.returncode==0,gpu.stderr
 audit={'utc':now(),'status':'completed_bounded224_448_resolution_comparison','primary_conclusion':'448 does not solve small-finger localization or larger-wrist generalization under fixed budget; no model adopted',
        'images':40,'states':20,'known_boxes':91,'new_annotations':0,'new_fits':2,'new_fit_resolution':448,'reused224_fits':2,'updates_each':600,
        '224_strict_reuse_and_exact_prediction_gate':True,'sampling_each_training_image_visits':300,'batch_images':8,
        'negative_patch_mass_control':'(256-P)/(G*G-P) per image/class; positive weights unchanged; old224 exact arithmetic; constant-logit focal and gradient total checked',
        'primary_AB_TP_FP_FN':{'224_seed0_front':[4,0,4],'224_seed1_front':[3,0,5],'448_seed0_front':[4,0,4],'448_seed1_front':[3,0,5],
         '224_seed0_finger':[5,5,24],'224_seed1_finger':[2,6,27],'448_seed0_finger':[3,9,26],'448_seed1_finger':[5,1,24]},
        'train_finger_TP_GT_448':[[25,30],[28,30]],'AB_static_finger_TP_GT_448':[[0,13],[1,13]],'AB_wrist_finger_TP_GT_448':[[3,16],[4,16]],
        'interpretation':'Train regression fits well but center-score misses remain; heldout loss is much larger. Grid change also alters physical3x3peak neighborhood, no isolated semantic-resolution causality claim.',
        'DINO_T_SupCon_updates':0,'not_visible_examples':0,'all20wrist_front_unknown':True,'unknowns_not_negative_or_scored':True,
        'historically_seen_data_not_independent_test':True,'assistant_repeat_not_independent_annotation_truth':True,
        'new_data_or_model_downloads':False,'third_resolution_or_network':False,'relation_stage':False,'model_adopted':False,'overall_guide_completed':False,
        'next_step_suggestion_only':'Independently review failed boxes and single-center target definition before any separate center-supervision study; not executed',
        'figures_visually_reviewed':['recall_train_development.png','finger_by_camera.png','examples_static.png','examples_wrist.png'],
        'report_only_final_edit':'Fixed identical image limits and described first failed development state; no inference/training or metrics changed',
        'links_checked':links,'frozen_phase_checks':phases,'gpu_compute_processes_at_seal':gpu.stdout.strip().splitlines(),
        'resource_or_permission_blockers':[]}
 save(E/'results/completion_audit.json',audit)
 save(E/'logs/seal_checks.json',{'utc':now(),'passed':True,'links_checked':links,'phase_manifest_checks':phases,'no_extra_optimizer_runs':True})
 files=sorted(p for base in[E,R]for p in base.rglob('*')if p.is_file()and p!=dest);hashes={str(p):sha(p)for p in files};sizes={str(p):p.stat().st_size for p in files}
 save(dest,{'utc':now(),'status':'sealed','files':len(files),'bytes':sum(sizes.values()),'sha256':hashes,'sizes':sizes,'manifest_excludes_itself':True,'scope':'Same40RGB,224reused/448twofixedseeds600steps; localization not reliable, no adoption'})
 check_manifest(dest)
 print('RESOLUTION_COMPARISON_SEALED',len(files),sum(sizes.values()),sha(dest),flush=True)
if __name__=='__main__':main()
