from common import *
import shutil,subprocess,torch
from fixed_dataset import CoreEpisodes
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import batch_indices

def main():
 assert not B.exists() and not REPORT.exists(),'Inspect and reuse existing run rather than overwrite'
 for d in ['source','configs','results','logs']:(B/d).mkdir(parents=True)
 (REPORT/'figures').mkdir(parents=True)
 for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,B/'source'/f.name)
 for name,source in [('fixed_design.json','proposed_protocol.json'),('measurement_addendum.json','proposed_measurement_addendum.json')]:shutil.copy2(DESIGN/'configs'/source,B/'configs'/name)
 guard('prepare')
 oldverify=subprocess.run([sys.executable,str(P/'experiments/rssm_encoder_gradient_20261019/source/verify_sealed.py')],capture_output=True,text=True,check=True)
 designverify=subprocess.run([sys.executable,str(DESIGN/'source/verify_delivery.py'),'--verify-seal'],capture_output=True,text=True,check=True)
 members=read(B/'configs/fixed_design.json')['members'];tr20=CoreEpisodes(P,'train',roots_override=members['train20']);norm=tr20.fit_stats()
 assert norm['fit_root_ids']==sorted(members['train20'])and norm['fit_transitions']==240
 save(B/'configs/common20_normalization.json',norm)
 feasibility=read(DESIGN/'results/design_feasibility.json')
 save(B/'results/coverage_review.json',{'utc':now(),'coverage':feasibility['coverage'],'common20_boundary_class_counts':norm['class_counts'],'common20_class_weights':norm['class_weights'],'valid_comparison':True,'limitations':['common20 has no slider change or light variation, no objectclass4/5/7, no intermediate drawer class1; cannot infer these capabilities','fixedcommon20 preprocessing andprobe labels controls these budgets; not ordinary fullpipeline data scaling','allroots historically inspected; heldout only with respect to currentoptimization','Ccombines scene andinitialheld-object order changes; externaldiagnostic only'],'no_root_selection_by_performance':True,'old_seal_before':oldverify.stdout,'design_seal_before':designverify.stdout})
 old=read(P/'experiments/rssm_encoder_gradient_20261019/source/protocol.json')
 for size in [20,40]:
  cell=B/'cells'/f'roots{size}'
  for d in ['configs','models','data','results','logs']:(cell/d).mkdir(parents=True)
  (REPORT/f'roots{size}').mkdir()
  protocol={'utc':now(),'fixed_design_sha256':sha(B/'configs/fixed_design.json'),'measurement_addendum_sha256':sha(B/'configs/measurement_addendum.json'),'size':size,'train_roots':members['train'+str(size)],'normalization_roots':members['train20'],'probe_roots':members['train20'],'validation_roots':members['validation10'],'test_roots':members['test10'],'externalC_roots':members['C20'],'architecture':old['architecture'],'gradient':old['gradient'],'training':{**old['training'],'normalization':'fixed common20 only','fits':4},'linear_probe':{**old['linear_probe'],'train_records':300,'dev_records':None,'sampling':'common20 fixed root/boundary pool, same600x32 materialized schedule acrosssizes andarms'},'evaluation':{'primary':'ABtest10 roots; fourfieldNLL andrelation9_exact forprobe/free3; seeds separately','modes':['current','one_step','wrong_action','persistence','free1','free3','free4'],'latent_samples':8,'no_fullprefix_claim':True,'bootstrap_roots':2000,'threshold':0.5,'final6000only':True,'all8models_andprobes_frozen_before_ABtest_orC':True,'splits':['train','common20','validation','test','externalC']},'no_new_collection':True,'no_D':True,'no_warmstart':True}
  save(cell/'configs/protocol.json',protocol);save(cell/'configs/normalization.json',norm)
  # Data order is deterministic root/branch sort, and all sibling branches remain together.
  episodes=sorted(e['episode_id']for e in CoreEpisodes(P,'train',roots_override=members['train'+str(size)]).episodes) if size==20 else sorted(r+'_b'+str(b)for r in members['train40']for b in range(3))
  save(cell/'configs/episode_order.json',episodes)
  class Dummy:pass
  d=Dummy();d.episodes=episodes;d.matched=list(range(len(episodes)))
  for seed in [0,1]:
   torch.manual_seed(seed);m=WorldModel(skill_count=6,event_count=3);partition(m);torch.save({'state':m.state_dict(),'seed':seed,'hash':th(m.state_dict()),'fresh_random_initialization':True},cell/f'models/initial_seed{seed}.pt');del m
   rng=np.random.default_rng(seed);indices=np.array([[batch_indices(d,rng).tolist()for _ in range(2)]for _ in range(6000)],np.int64);latent=np.arange(12000,dtype=np.int64).reshape(6000,2)+261019000+seed*20000
   np.savez_compressed(cell/f'configs/schedule_seed{seed}.npz',indices=indices,latent_seeds=latent)
 refs=[P/'src/wm_symbolic'/n for n in ['world_model.py','dataset.py','checkpointing.py']]
 refs += [DESIGN/'configs/delivery_manifest.json',B/'configs/fixed_design.json',B/'configs/measurement_addendum.json',P/'data/core1200_v1/data_manifest.json',P/'data/core1200_v1/roots_manifest.json',P/'data/core1200_v1/features/index.json',P/'runs/core1200_v1/protocol.json']
 ids=set(members['train40'])
 refs +=[Path(r['labels_path'])for r in read(P/'data/core1200_v1/data_manifest.json')['records']if r['root_id']in ids]
 refs +=[Path(r['path'])for r in read(P/'data/core1200_v1/features/index.json')['items']if r['root_id']in ids]
 save(B/'configs/references.json',{'sha256':{str(p):sha(p)for p in refs}})
 save(B/'results/preparation.json',{'utc':now(),'newfits':0,'new_inference':0,'shared_normalization_sha256':sha(B/'configs/common20_normalization.json'),'fixed_memberships_sha256':sha(B/'configs/fixed_design.json'),'original_design_sha256':sha(DESIGN/'configs/proposed_protocol.json'),'old_weights_loaded':False})
 print('PREPARED',B,flush=True)

if __name__=='__main__':
 import sys
 main()
