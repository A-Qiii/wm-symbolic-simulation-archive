from common import *
from score import score
import subprocess,sys

def main():
 # Read only an already sealed historical C prediction, never a new model/test prediction.
 old=P/'experiments/rssm_encoder_gradient_20261019/data/predictions_joint_seed0_dev.npz'
 a=dict(np.load(old));mask=a['mode']=='free3';a={k:v[mask]for k,v in a.items()}
 meta={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')if r['environment']in ['A','B','C']}
 rows,rr,rel=score(a,{'size':60,'seed':0,'arm':'joint','split':'historicalC_only'},meta)
 def take(f,metric):return next(r for r in rows if r['fact']==f and r['metric']==metric and r['stratum']=='all'and r['change']=='all')
 oldscore=read(P/'experiments/rssm_encoder_gradient_20261019/results/scores.json')
 expected=next(r['value']for r in oldscore['primary']if r['arm']=='joint'and r['seed']==0 and r['split']=='dev'and r['mode']=='free3'and r['metric']=='NLL')
 assert take('primary4','NLL')['value']==expected
 assert round(take('relation9_exact','accuracy')['micro']*60)==51
 assert round(take('all5_exact','accuracy')['micro']*60)==12
 records=read(P/'data/core1200_v1/data_manifest.json')['records'];members=read(B/'configs/fixed_design.json')['members'];checks=[]
 for size in [20,40]:
  for rid in [members['test10'][0],members['validation10'][0],members['C20'][0],'D_core1200_00']:
   path=next(r['labels_path']for r in records if r['root_id']==rid)
   code='from common import *\nguard("train")\nfrom pathlib import Path\nPath('+repr(path)+').read_bytes()'
   env=os.environ.copy();env['RSSM_SCALE_ROOTS']=str(size)
   out=subprocess.run([sys.executable,'-c',code],cwd=B/'source',env=env,capture_output=True,text=True)
   assert out.returncode!=0 and 'Root access forbidden' in out.stderr,(size,rid,out.stderr)
   checks.append({'size':size,'root':rid,'read_denied_before_open':True})
 save(B/'results/score_and_guard_checks.json',{'utc':now(),'passed':True,'historical_C_free3_primaryNLL_exact':expected,'historical_C_free3_relation9_correct':51,'historical_C_free3_all5_correct':12,'guard_checks':checks,'new_model_predictions':0,'current_test_predictions':0,'current_test_samples_read':0})
 print('SCORE_AND_GUARD_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
