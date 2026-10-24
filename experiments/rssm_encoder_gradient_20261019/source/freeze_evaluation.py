from common import *
def main():
 verify();guard('freeze_eval');assert not list((E/'data').glob('predictions_*.npz'))
 addendum={'utc':now(),'training_protocol_changed':False,'main_fits_or_budget_changed':False,'primary_metric_changed':False,'before_any_saved_model_prediction':True,'clarification':'For free1/free3/free4, also report root_persistence1/3/4: copy the same inferred root posterior symbols to the same real endpoint. This supplements the already declared one-step persistence control without fitting, collecting, or using future input.','metrics':'NLL clipped at1e-9 for numerical safety. Primary is mean of3object+drawer NLL within state then root mean. Other accuracy/NLL root-macro; precision/recall/BA pooled ratios with entire-root cluster bootstrap. Undefined denominators remain null.','guard_limit':'Python audit-hook policy, not an OS-level sandbox; project data paths are checked, and actual scripts use only core A/B/C inputs.','no_support_stratum':'No bilateral+table-supported endpoint denominator, do not label absent strata as zero FPR.'}
 save(E/'configs/evaluation_addendum.json',addendum)
 files=[E/'source'/n for n in ['evaluate.py','probes.py','score.py','test_score.py','freeze_evaluation.py','scope_design.py','common.py']]+[E/'configs/evaluation_addendum.json']
 save(E/'configs/evaluation_frozen.json',{'utc':now(),'before_any_saved_model_prediction':True,'sha256':{str(p):sha(p)for p in files}})
 for p in files:p.chmod(0o444)
 print('evaluation sources frozen',sha(E/'configs/evaluation_frozen.json'))
if __name__=='__main__':main()
