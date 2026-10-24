from common import *
def main():
 assert not(S/'configs/design_frozen.json').exists();budget_check()
 for p,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h
 refs=list((S/'source').glob('*.py'))+list((S/'configs').glob('*.json'))+[P/f'src/wm_symbolic/{n}.py'for n in ['physical_oracle','env_adapter','core_ownership','calibrated_labels','world_model','dataset','vision_cache']]+[P/'scripts/evaluate_physical_oracle.py',CELL/'configs/normalization.json']+[CELL/f'models/joint_seed{s}/latest.pt'for s in [0,1]]
 historical={}
 for n in ['failure_recovery_control_20261019','initial_state_audit_20261019','paired_outcome_training_20261019','observation_state_update_20261019','observation_retention_audit_20261019','balanced_branches_20261019']:
  d=P/'experiments'/n;historical[str(d)]=read(d/'SEALED.json')
 save(S/'configs/historical_seals_before.json',historical)
 save(S/'configs/design_frozen.json',{'utc':now(),'before_collection':True,'sha256':{str(p):sha(p)for p in refs},'actual_geometries_and_all_role_assignments_frozen':True});print('DESIGN_FROZEN',len(refs),flush=True)
if __name__=='__main__':main()
