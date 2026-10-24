from config import *
import ast,shutil
def main():
 assert not(E/'configs/frozen.json').exists();refs=read(E/'configs/references.json')['sha256'];cfg=read(E/'configs/protocol.json');cfg['prefix_replacement']='If at least one independent form passes the prespecified actual gate: exactly one fixed intervention on all11 complete normal-branch roots, replace after-lift z with saved actual z; predict place then close without another correction. Both mixed/alphaZ, both seeds. Compare Holding after each; empty-close branch omitted because one-step already wrong.';save(E/'configs/protocol.json',cfg)
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text());shutil.copy2(f,E/'source'/f.name)
 for f in E.rglob('*'):
  if f.is_file():refs[str(f)]=sha(f)
 for f in [D/'configs/protocol.json',REP/'source/model.py',WHOLE/'configs/query_normalization.npz']:
  refs[str(f)]=sha(f)
 for v in read(D/'configs/protocol.json')['models'].values():refs[v['path']]=sha(v['path'])
 for v in read(D/'configs/protocol.json')['query_heads'].values():refs[v['original']['path']]=sha(v['original']['path'])
 save(E/'configs/frozen.json',{'utc':now(),'before_any_optimizer_updates':True,'max_fits':4,'updates_each':600,'sha256':refs});print('HOLDING_PROTOCOL_FROZEN',len(refs),'refs')
if __name__=='__main__':main()
