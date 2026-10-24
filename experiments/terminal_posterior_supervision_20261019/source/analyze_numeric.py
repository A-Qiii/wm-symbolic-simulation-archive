from common import *
import analyze
def main():
 guard('eval');out=S/'results/numerical_attempt1';assert not out.exists();out.mkdir();old={}
 for name in ['metrics.csv','current_pairs.csv','assimilation.csv','memory.csv','false_relations.csv','internal_changes.csv','cases.json','analysis_complete.json']:
  p=S/'results'/name;old[name]=sha(p);p.rename(out/name)
 p=S/'data/analysis_records.json';old['analysis_records.json']=sha(p);p.rename(out/'analysis_records.json')
 save(S/'configs/numerical_amendment.json',{'utc':now(),'before_rescoring':True,'reason':'Float32 upper clipping bound1-1e-9 rounds to1, causing0*log0 NaNs in offline NumPy NLL.','sole_change':'Promote offline NLL inputs tofloat64 before same clipping and same equation. All accuracy metrics, thresholds, training loss, dev selection, model weights, predictions and source analyze.py unchanged.','original_outputs_sha256':old,'wrapper_sha256':sha(__file__),'new_forwards':0,'new_training':0})
 original=analyze.nll
 def finite_nll(p,y):
  x=original(np.asarray(p,dtype=np.float64),np.asarray(y,dtype=np.float64));assert np.isfinite(x).all();return x
 analyze.nll=finite_nll;analyze.main()
 before=list(csv.DictReader((out/'metrics.csv').open()));after=list(csv.DictReader((S/'results/metrics.csv').open()));assert len(before)==len(after)
 for a,b in zip(before,after):assert {k:v for k,v in a.items()if k!='mean_nll'}=={k:v for k,v in b.items()if k!='mean_nll'}
 for name in ['current_pairs.csv','assimilation.csv','memory.csv','false_relations.csv','internal_changes.csv']:assert sha(S/'results'/name)==old[name],name
 save(S/'results/numerical_fix_verification.json',{'utc':now(),'all_metric_rows_accuracy_CI_unchanged':len(after),'all_NLL_finite':True,'five_other_metric_files_byte_identical':True,'no_training_or_forwards':True});print('NUMERICAL_FIX_VERIFIED',len(after),flush=True)
if __name__=='__main__':main()
