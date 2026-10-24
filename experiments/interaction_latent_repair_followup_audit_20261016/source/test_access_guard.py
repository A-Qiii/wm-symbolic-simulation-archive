import tempfile,os,json,subprocess,sys
from pathlib import Path
from access_guard import TrainingAccessPolicy
def main():
 records=[]
 with tempfile.TemporaryDirectory(prefix='wm_guard_test_')as tmp:
  p=Path(tmp);run=p/'experiments/run';(run/'source').mkdir(parents=True);(run/'source/train.py').write_text('');(run/'data/public').mkdir(parents=True);policy=TrainingAccessPolicy(p,run,'canonical_control',0)
  cases=[('AB input',run/'data/public/inputs.npz','r',True),('C pair metadata',run/'data/public/pairs.json','r',True),('protocol',run/'configs/protocol.json','r',True),('runtime import outside project',Path('/usr/lib/python3.11/pathlib.py'),'r',True),('own checkpoint',run/'models/canonical_control_seed0/update0250.pt','wb',True),('own report',run/'results/train_canonical_control_0.json','w',True),('D2 normalized',run/'data/D2/public/inputs.npz','r',False),('D2 raw',run/'data/D2_raw/root/public/x.npz','r',False),('old D public',p/'experiments/interaction_latent_query_20261016/data/D/public/inputs.npz','r',False),('old rootdata test',p/'data/legacy/D/test.npz','r',False),('private labels',run/'data/_private/labels.json','r',False),('parent traversal',run/'data/public/../../data/D2_raw/x.npz','r',False),('overwrite input',run/'data/public/inputs.npz','w',False),('another arm overwrite',run/'models/canonical_matched_seed0/update0250.pt','w',False),('raw fd',99,'r',False)]
  hidden=p/'experiments/old_D/public.npz';hidden.parent.mkdir(parents=True);hidden.write_text('DO NOT READ');alias=run/'data/public/inputs.npz';alias.symlink_to(hidden)
  # The symlink is installed after the normal allow case, so use pairs.json for its positive counterpart.
  cases[0]=('AB pair allow',run/'data/public/pairs.json','r',True);cases.append(('allowlisted filename symlink to old D',alias,'r',False));cases.append(('bytes D2 path',os.fsencode(run/'data/D2_raw/x.npz'),'r',False))
  outside=p/'outside_protected_roots.npz';outside.write_text('NO');outside_alias=run/'data/public/normalization.npz';outside_alias.symlink_to(outside);cases.append(('symlink escape outside protected roots',outside_alias,'r',False))
  for name,path,mode,expected in cases:
   try:actual=policy.check(path,mode)
   except PermissionError:actual=False
   assert actual==expected,(name,actual,expected);records.append({'case':name,'allowed':actual,'expected':expected,'passed':True})
  # Exercise the real audit callback in an isolated interpreter, not only its predicate.
  code='''import sys,json
from pathlib import Path
from access_guard import install_training_guard
p=Path(sys.argv[1]);r=p/'experiments/run';log=[];install_training_guard(p,r,'canonical_control',0,log)
try:
 open(r/'data/D2_raw/leak.npz','rb')
except PermissionError:
 print('AUDIT_HOOK_BLOCKED_BEFORE_FILE_LOOKUP')
else:raise AssertionError('D2 raw not blocked')
'''
  z=subprocess.run([sys.executable,'-c',code,str(p)],capture_output=True,text=True,env=os.environ|{'PYTHONPATH':str(Path(__file__).parent)});assert z.returncode==0 and'AUDIT_HOOK_BLOCKED'in z.stdout,(z.stdout,z.stderr)
  records.append({'case':'isolated real sys.addaudithook open','passed':True})
 print(json.dumps({'passed':True,'cases':len(records),'records':records,'scope':'future Python open guard; native I/O and inherited descriptors require OS-level isolation'},indent=2))
if __name__=='__main__':main()
