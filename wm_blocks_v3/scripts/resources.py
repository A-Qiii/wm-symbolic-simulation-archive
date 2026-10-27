"""Explicit external-resource fetch or read-only verification. Never pushes."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paths import ROOT,CALVIN_ENV,DINO_WEIGHTS
def manifest():return json.loads((ROOT/'configs/external_resources.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_calvin():
 q=manifest()['calvin_env']
 head=subprocess.check_output(['git','-C',str(CALVIN_ENV),'rev-parse','HEAD'],text=True).strip()
 if head!=q['commit']:raise RuntimeError('CALVIN commit differs from pinned version')
 dirty=subprocess.check_output(['git','-C',str(CALVIN_ENV),'status','--porcelain','--untracked-files=no'],text=True)
 if dirty:raise RuntimeError('CALVIN tracked files have local changes; keep them and use a separate clean checkout')
 for n,h in q['files_sha256'].items():
  if digest(CALVIN_ENV/n)!=h:raise RuntimeError('CALVIN resource mismatch: '+n)
 return {'resource':'calvin_env','commit':head,'key_files':len(q['files_sha256'])}
def verify_dino():
 q=manifest()['dinov2']
 for n,r in q['files'].items():
  p=DINO_WEIGHTS/n
  if p.stat().st_size!=r['bytes']or digest(p)!=r['sha256']:raise RuntimeError('DINO resource mismatch: '+n)
 return {'resource':q['repo_id'],'revision':q['revision'],'files':len(q['files'])}
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch-calvin','fetch-dino','verify']);a=p.parse_args()
 if a.action=='fetch-calvin':
  q=manifest()['calvin_env']
  if not CALVIN_ENV.exists():
   CALVIN_ENV.parent.mkdir(parents=True,exist_ok=True)
   subprocess.run(['git','clone','--no-checkout',q['git_url'],str(CALVIN_ENV)],check=True)
   subprocess.run(['git','-C',str(CALVIN_ENV),'checkout','--detach',q['commit']],check=True)
  print(json.dumps(verify_calvin()))
 elif a.action=='fetch-dino':
  q=manifest()['dinov2']
  if not DINO_WEIGHTS.exists():
   # This branch is reached only after the explicit fetch-dino command.
   os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id=q['repo_id'],revision=q['revision'],local_dir=str(DINO_WEIGHTS),allow_patterns=list(q['files']))
  print(json.dumps(verify_dino()))
 else:print(json.dumps({'verified':[verify_calvin(),verify_dino()],'network_used':False}))
if __name__=='__main__':main()
