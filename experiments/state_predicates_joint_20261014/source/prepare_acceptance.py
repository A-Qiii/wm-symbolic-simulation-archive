from pathlib import Path
import json,hashlib,datetime
import numpy as np
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_joint_20261014';B=P/'data/predicate_joint_acceptance_20261014';C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    assert not B.exists() and not (E/'results/candidate_freeze.json').exists()
    result=read(E/'results/balanced_loss.json');assert result['qualified_count']>=2
    protocol=read(E/'configs/protocol.json')
    for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
    models=[{'seed':r['seed'],'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256'],'thresholds':r['calibration']['thresholds'],'qualified':r['qualified']}for r in result['seeds']]
    freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'family':'balanced_loss','models':models,'qualified_seeds':[r['seed']for r in models if r['qualified']],'all_three_seeds_reported':True,'source_sha256':{str(p):sha(p)for p in (E/'source').glob('*.py')},'protocol_sha256':sha(E/'configs/protocol.json'),'acceptance_data_absent':not B.exists(),'no_acceptance_based_selection':True}
    save(E/'results/candidate_freeze.json',freeze);(E/'results/candidate_freeze.json').chmod(0o444)
    old=[]
    for f in [P/'data/core1200_v1/roots_manifest.json',P/'runs/proposition_interface_20261014/roots_manifest.json',P/'data/predicate_gap_supplement_20261014/roots_manifest.json']:
        for r in read(f):
            if r['environment']=='D' and 'layout' in r:old.append((np.array([r['layout']['block_'+c]for c in C]),r['root_id']))
    cfg=OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml');names=list(cfg.objects.movable_objects)
    paths=list((P/'data/roots').glob('*/D_*/reset_evaluator_only.npz'))+[q for v in ['v1','v2','v3']for q in (P/f'runs/physical_oracle_{v}/roots').glob('D_*/reset_evaluator_only.npz')]
    for f in paths:
        with np.load(f)as z:a=z['scene_obs'][6:].reshape(3,6);lookup=dict(zip(names,a));old.append((np.array([lookup['block_'+c]for c in C]),str(f)))
    nprior=len(old);lo,hi=np.array(cfg.surfaces.table);roots=[];ex=[]
    for ci,color in enumerate(C):
        for di,drawer in enumerate(['closed','open']):
            for rep in range(2):
                i=ci*4+di*2+rep;seed=10142600+i;rng=np.random.default_rng(seed)
                for attempt in range(10000):
                    poses=np.zeros((3,6));order=rng.permutation(3)
                    for slot,j in enumerate(order):
                        poses[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.02,.02),rng.uniform(-.135,-.05),.46]
                        poses[j,5]=[-np.pi/3,0,np.pi/3][(ci+di+rep+j)%3]+rng.uniform(-.12,.12)
                    nearest=[]
                    for other,name in old:
                        dist=np.linalg.norm(poses[:,:3]-other[:,:3],axis=1);yaw=np.abs((poses[:,5]-other[:,5]+np.pi/2)%np.pi-np.pi/2)
                        if np.all(dist<=.020) or(np.all(dist<=.030)and np.all(yaw<=.25)):nearest.append(name)
                    if not nearest:break
                else:raise RuntimeError('No excluded-safe layout under finite proposal budget')
                rid=f'D_joint_accept_{color}_{drawer}_{rep}';root={'root_id':rid,'environment':'D','split':'acceptance','seed':seed,'target_color':color,'drawer_setting':drawer,'drawer_reset_m':0 if drawer=='closed'else .22,'layout':{'block_'+c:poses[j].tolist()for j,c in enumerate(C)},'near_probe_yaw_offset_rad':0 if(ci+di+rep)%2==0 else np.pi/2,'near_probe_translation_m':[0,-.055,.025],'follow_probe_translation_m':[.015 if rep==0 else -.015,0,0],'trajectories':['grasp','near_closed_empty']+(['place']if drawer=='open'else[])}
                roots.append(root);ex.append({'root_id':rid,'rejected_proposals':attempt,'prior_comparison_count':len(old)});old.append((poses,rid))
    for d in [B/'episodes',B/'roots',E/'data/features']:d.mkdir(parents=True,exist_ok=True)
    save(B/'roots_manifest.json',roots);(B/'roots_manifest.json').chmod(0o444)
    save(B/'data_manifest.json',{'schema':'predicate_joint_acceptance_v1','records':[],'trajectories':[]})
    save(E/'configs/acceptance_roots.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_manifest_sha256':sha(B/'roots_manifest.json'),'prior_D_roots_compared':nprior,'rows':ex,'candidate_freeze_sha256':sha(E/'results/candidate_freeze.json'),'source_sha256':{str(p):sha(p)for p in (E/'source').glob('*.py')},'root_count':12,'trajectory_count':30,'no_outcome_resampling':True})
    (E/'configs/acceptance_roots.json').chmod(0o444);print('CANDIDATES_AND_ROOTS_FROZEN',[r['seed']for r in models if r['qualified']],len(roots),flush=True)
if __name__=='__main__':main()
