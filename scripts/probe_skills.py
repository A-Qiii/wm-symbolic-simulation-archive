"""Stage 0 real HULC evaluation on C, isolated from D test roots.

All 25 skills stay in the catalogue. Representative family tests compare fixed
120/240/360-step prefixes of the same seeded rollout. No success-driven stop.
"""
from pathlib import Path
import argparse,json,random,sys,time,traceback
import numpy as np
import torch,hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed

P=Path('/home/robot/research/projects/wm_symbolic')

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--checkpoint',required=True);pa.add_argument('--train-folder',required=True)
    pa.add_argument('--embeddings',required=True);pa.add_argument('--seeds',type=int,default=3);a=pa.parse_args()
    out=P/'runs'/time.strftime('stage0_skills_%Y%m%d_%H%M%S');out.mkdir()
    annotations=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
    task_cfg=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml')
    families={k:('drawer' if k in ['open_drawer','close_drawer'] else 'slider' if k.startswith('move_slider') else
                 'lift' if k.startswith('lift_') else 'place' if k.startswith('place_') else
                 'device' if k.startswith('turn_') else 'push') for k in annotations
              if k in ['open_drawer','close_drawer','move_slider_left','move_slider_right','place_in_drawer','place_in_slider']
              or k.startswith(('lift_','turn_')) or (k.startswith('push_') and k!='push_into_drawer')}
    assert len(families)==25
    catalogue=[{'skill':k,'family':v,'command':annotations[k][0],'status':'untested'} for k,v in families.items()]
    (P/'configs/skills.json').write_text(json.dumps(catalogue,indent=2))
    evaluator=hydra.utils.instantiate(task_cfg)
    policy=FrozenHulc(a.train_folder,a.checkpoint,a.embeddings)
    (out/'executor_identity.json').write_text(json.dumps(policy.identity,indent=2))
    # No GT-derived action filtering. This list is fixed before observing roots.
    representatives=['open_drawer','move_slider_left','lift_red_block_table','turn_on_lightbulb','push_blue_block_left']
    results=[]
    for seed in range(a.seeds):
        env=CalvinAdapter(P,'C',seed,tabletop=True)
        try:
            obs=env.reset()
            for _ in range(12):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
            root=env.evaluator_snapshot()
            for skill in representatives:
                env.reset(root);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                r=execute_fixed(env,policy,skill,annotations[skill][0],360,out/f'C_root{seed:03d}_{skill}',evaluator)
                r.update(seed=seed,family=families[skill]);results.append(r)
                (out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
        finally:env.close()
    counts={fam:{str(b):sum(r['score_at_fixed_budgets'][str(b)] for r in results if r['family']==fam)
                  for b in [120,240,360]} for fam in set(families[s] for s in representatives)}
    budgets={fam:min([120,240,360],key=lambda b:(-counts[fam][str(b)],b)) for fam in counts}
    (out/'budget_comparison.json').write_text(json.dumps({'counts':counts,'denominator_per_family':a.seeds,
        'candidate_budget_steps':[120,240,360],'selected':budgets,
        'selection':'Highest final-prefix success count, then shortest budget; raw failures retained. Not yet a stability claim.'},indent=2))
    # Reference sequence is evaluation-only; never given to the learned planner.
    chain=['open_drawer','close_drawer','move_slider_left','turn_on_lightbulb']
    env=CalvinAdapter(P,'C',0,tabletop=True);chain_results=[]
    try:
        env.reset()
        for _ in range(12):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
        torch.manual_seed(0)
        for i,skill in enumerate(chain):
            r=execute_fixed(env,policy,skill,annotations[skill][0],budgets[families[skill]],out/f'chain_{i}_{skill}',evaluator)
            chain_results.append(r)
    finally:env.close()
    summary={'family_successes':{f:max(c.values()) for f,c in counts.items()},'budget_comparison':counts,
             'reference_chain':chain,'chain_results':chain_results,
             'chain_complete':all(r['score_at_fixed_budgets'][str(r['requested_steps'])] for r in chain_results),
             'test_D_roots_used':False,'training_transitions_added':0}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
