from common import *
import time,fcntl,traceback
from collect import Runtime,ForkInvalid,compare

def main():
 assert not(Q/'results/started.json').exists()
 for name in ['frozen.json','input_references.json']:
  for p,h in read(E/'configs'/name)['sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(Q/'results/started.json',{'utc':now()});protocol=read(E/'configs/protocol.json');spec=protocol['roots'][0];rt=None;stop=None;results=[];t=time.perf_counter();preparation_controls=None
 try:
  rt=Runtime.__new__(Runtime);rt.__init__(spec,protocol,output_root=Q,qualification=True)
  prepared=rt.prepare_condition('aligned');preparation_controls=rt.total_controls
  if prepared is None:raise ForkInvalid('qualification_preparation_failed_no_replacement')
  stored={}
  for item in read(Q/'configs/protocol.json')['runs']:
   ref=stored.get(item['compare_to']);result=rt.execute_branch(item['action'],item['name'],prepared,ref,item['compare_controls']);stored[item['name']]=result;results.append(result['row']);save(Q/'results/runs.json',results)
   if result['row']['status']!='completed':raise ForkInvalid('qualification_program_incomplete:'+item['name'])
   print('QUALIFICATION_RUN',item['name'],result['row']['status'],result['row']['executed_controls'],flush=True)
  checks=[c for result in stored.values()for c in result['checks']]
  assert len(results)==4 and sum(x['executed_controls']for x in results)==288
  passed=all(c['equal']for c in checks)and all(x['symmetric_live_restore']for x in results)
  save(Q/'results/QUALIFICATION.json',{'utc':now(),'passed':passed,'scope':'same-process live native S; every branch including first restored; no historical-online or file-reload estimand','runs':4,'branch_controls':288,'preparation_controls':preparation_controls,'comparisons':len(checks),'failed_comparisons':sum(not c['equal']for c in checks),'same_H_controls':72,'same_L_controls':72,'H_L_common_controls':12,'first_control_substeps_per_pair':8,'paired_start_checks':3,'contact_multiset_only':True,'tolerance':0,'old_failed_qualification_unchanged':True})
  if not passed:stop={'kind':'qualification_mismatch','utc':now()}
 except Exception as exc:
  stop={'kind':'qualification_failed','exception':repr(exc),'traceback':traceback.format_exc(),'utc':now()}
 finally:
  if stop:
   save(Q/'results/STOPPED.json',stop)
   if not(Q/'results/QUALIFICATION.json').exists():save(Q/'results/QUALIFICATION.json',{'utc':now(),'passed':False,'stop':stop,'scope':'same-process symmetric live native S only'})
  allrows=rt.branches if rt else []
  summary={'utc':now(),'completed_programs':len(results),'attempted_programs':len(allrows),'branch_controls':sum(x.get('executed_controls',0)for x in allrows),'preparation_controls':preparation_controls,'all_controls_including_preparation':rt.total_controls if rt else 0,'actual_ticks':rt.actual_ticks if rt else 0,'stop':stop,'seconds':time.perf_counter()-t,'fits':0,'new_distinct_layouts':0,'files_reloaded_into_physics':0}
  save(Q/'results/execution_summary.json',summary)
  if rt:rt.close()
  lock.close()
 print('QUALIFICATION_END',summary,flush=True)

if __name__=='__main__':main()
