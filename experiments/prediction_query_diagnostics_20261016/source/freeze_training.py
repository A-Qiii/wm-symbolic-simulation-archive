from config import *
def main():
 assert not list((E/'models').glob('*')), 'Freeze must precede query training'
 cfg=read(E/'configs/head_protocol.json');sel=read(E/'configs/head_selection.json');coverage=read(E/'results/head_data_coverage.json');draws={}
 packs={name:dict(np.load(E/'data'/('evaluator'if name=='C_validation'else'head_train')/(name+'.npz')))for name in ['original_coverage','intermediate_coverage','C_selection','C_validation']}
 assert not(set(packs['C_selection']['root_ids'])&set(packs['C_validation']['root_ids']))
 for seed in[0,1]:
  schedules={arm:dict(np.load(E/'configs'/f'head_schedule_{arm}_seed{seed}.npz'))for arm in cfg['arms']}
  assert np.array_equal(schedules['original_coverage']['state_ids'][:,:20],schedules['intermediate_coverage']['state_ids'][:,:20])
  for arm,s in schedules.items():
   d=packs[arm];assert np.array_equal(d['state_ids'][s['indices']],s['state_ids']);assert s['indices'].size==38400;assert len(np.unique(s['indices']))==144
   v=d['v'][s['indices']];draws[f'{arm}_seed{seed}']={'draws':int(s['indices'].size),'distinct_states':len(np.unique(s['indices'])),'valid_fact_supervision_draws':int(v.sum()),'full9_draws':int(v.all(-1).sum()),'partial_draws':int((~v.astype(bool).all(-1)).sum())}
 paths=list((E/'configs').glob('head*'))+[E/'source'/f for f in ['config.py','prepare_heads.py','encode_heads.py','train_head.py','freeze_training.py']]+[E/'data'/('evaluator'if name=='C_validation'else'head_train')/(name+'.npz')for name in packs]
 save(E/'configs/training_frozen.json',{'frozen_utc':now(),'models_present_before_freeze':0,'paired_base_draws_identical':True,'C_selection_validation_root_overlap':0,'exact_public_content_overlap':coverage['exact_public_content_split_overlap'],'sample_budgets':draws,'important_limit':'Equal sample/update budget does not mean equal label-information budget: new intermediate records have only6 valid labels. Coverage arm also adds4 AB roots; causal contrast is this coverage-and-sampling package, not isolated oversampling alone.','sha256':{str(p):sha(p)for p in sorted(paths)}})
 print('TRAINING_INPUTS_AND_SOURCE_FROZEN',json.dumps(draws),flush=True)
if __name__=='__main__':main()
