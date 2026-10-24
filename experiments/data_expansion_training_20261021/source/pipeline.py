from common import *
import torch,fcntl,traceback
import features,training,evaluation,scoring
def main():
 guard();verify();budget();lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 try:
  save(S/'results/stage.json',{'utc':now(),'stage':'train_development_cache','pid':os.getpid()});features.phase(False)
  save(S/'results/stage.json',{'utc':now(),'stage':'training'});training.phase()
  save(S/'results/stage.json',{'utc':now(),'stage':'development_evaluation'});evaluation.phase(False);scoring.phase()
  save(S/'results/stage.json',{'utc':now(),'stage':'frozen_test_cache'});features.phase(True)
  save(S/'results/stage.json',{'utc':now(),'stage':'frozen_test_evaluation'});evaluation.phase(True);scoring.phase()
  save(S/'results/pipeline_complete.json',{'utc':now(),'new_fits':2,'reused_old_fits':2,'updates_each':1500,'completed':True});print('PIPELINE_COMPLETE',flush=True)
 except BaseException as e:
  save(S/'results/STOP.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
