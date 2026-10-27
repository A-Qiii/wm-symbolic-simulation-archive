"""Explicitly create project-local environments, never change host drivers."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--executor-python',default='python3.10');p.add_argument('--model-python',default='python3.11');a=p.parse_args()
 for name,interpreter,version,requirements in [('executor',a.executor_python,'3.10','physics.txt'),('world_model',a.model_python,'3.11','model.txt')]:
  actual=subprocess.check_output([interpreter,'-c','import sys; print("%s.%s"%sys.version_info[:2])'],text=True).strip()
  if actual!=version:raise RuntimeError(name+' requires Python '+version)
  env=ROOT/'.envs'/name
  if env.exists():raise FileExistsError(str(env)+' already exists; preserve it and use a separate project checkout')
  subprocess.run([interpreter,'-m','venv',str(env)],check=True)
  subprocess.run([str(env/'bin/python'),'-m','pip','install','-r',str(ROOT/'requirements'/requirements)],check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/init_workspace.py')],check=True)
if __name__=='__main__':main()
