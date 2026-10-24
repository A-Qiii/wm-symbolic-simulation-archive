from common import *
def main():
 refs=read(B/'configs/references.json')['sha256']
 for p,h in refs.items():assert sha(p)==h,p
 for size in [20,40]:
  cell=B/'cells'/f'roots{size}'
  assert read(cell/'results/preflight.json')['passed']and read(cell/'results/preflight_bf16.json')['passed']
  assert not(cell/'configs/frozen.json').exists()
  files=list((B/'source').glob('*.py'))+[p for p in(B/'configs').glob('*.json')]+[p for p in(cell/'configs').glob('*')]+list((cell/'models').glob('initial_*.pt'))+[cell/'results/preflight.json',cell/'results/preflight_bf16.json']
  hashes={str(p):sha(p)for p in files};hashes.update(refs)
  save(cell/'configs/frozen.json',{'utc':now(),'before_any_fit':True,'sha256':hashes,'protocol_sha256':sha(cell/'configs/protocol.json')})
 print('FROZEN_BOTH_SIZES_BEFORE_ANY_FIT',flush=True)
if __name__=='__main__':main()
