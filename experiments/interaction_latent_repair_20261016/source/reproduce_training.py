"""Prepare an isolated AB/C-only reproduction. No physical collection or D2 copied."""
from common import *
import argparse,shutil,subprocess,sys
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True,type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();dest=a.destination.resolve();assert dest.is_relative_to(P/'experiments')and not dest.exists();dest.mkdir();(dest/'source').mkdir()
 for name in['common.py','model.py','runtime.py','train.py']:
  s=(E/'source'/name).read_text()
  if name=='common.py':s=s.replace("E=P/'experiments/interaction_latent_repair_20261016'",f'E=Path({str(dest)!r})').replace("R=P/'reports/interaction_latent_repair_20261016'",f'R=Path({str(dest / "report")!r})')
  (dest/'source'/name).write_text(s)
 for folder in['configs','models','data/public','logs','results']:(dest/folder).mkdir(parents=True,exist_ok=True)
 for name in['protocol.json','schedule_seed0.npz','schedule_seed1.npz']:shutil.copy2(E/'configs'/name,dest/'configs'/name)
 for seed in[0,1]:
  for prefix in['initial','query']:shutil.copy2(E/'models'/f'{prefix}_seed{seed}.pt',dest/'models'/f'{prefix}_seed{seed}.pt')
 for name in['inputs.npz','normalization.npz','pairs.json']:shutil.copy2(E/'data/public'/name,dest/'data/public'/name)
 files=[p for p in dest.rglob('*')if p.is_file()];save(dest/'configs/training_frozen.json',{'created_utc':now(),'copied_from_sealed_AB_C_artifacts':str(E),'no_D_data':True,'sha256':{str(p):sha(p)for p in files}})
 commands=[[sys.executable,str(dest/'source/train.py'),'--arm',arm,'--seed',str(seed)]for arm in['canonical_control','canonical_matched']for seed in[0,1]]
 save(dest/'configs/reproduction_commands.json',commands)
 if a.run:
  for cmd in commands:subprocess.run(cmd,check=True)
 print('AB_C_REPRODUCTION_READY',dest,'ran_training',a.run,'physical_or_D2_execution',False)
if __name__=='__main__':main()
