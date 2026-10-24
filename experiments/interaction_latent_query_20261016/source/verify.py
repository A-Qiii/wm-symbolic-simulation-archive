"""Read-only content verification of delivered experiment and linked sources."""
from pathlib import Path
import json,hashlib
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/interaction_latent_query_20261016';R=P/'reports/interaction_latent_query_20261016'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()
def main():
 m=json.loads((E/'configs/delivery_manifest.json').read_text());bad=[]
 for p,h in m['sha256'].items():
  if not Path(p).is_file()or sha(p)!=h:bad.append(p)
 assert not bad,bad
 print(json.dumps({'passed':True,'verified_files':len(m['sha256']),'bytes':m['bytes'],'read_only':True,'model_retraining_or_physical_execution':False}))
if __name__=='__main__':main()
