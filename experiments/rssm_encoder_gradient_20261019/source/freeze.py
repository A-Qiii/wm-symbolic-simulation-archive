from common import *
def main():
 assert read(E/'results/preflight.json')['passed']and read(E/'results/preflight_bf16.json')['passed'];assert not(E/'configs/frozen.json').exists()
 refs=read(E/'configs/references.json')['sha256']
 for p,h in refs.items():assert sha(p)==h,p
 files=[p for d in ['source','configs']for p in(E/d).rglob('*')if p.is_file()]+list((E/'models').glob('initial_*.pt'))+[E/'results/preflight.json',E/'results/preflight_bf16.json']
 hashes={str(p):sha(p)for p in files};hashes.update(refs)
 save(E/'configs/frozen.json',{'utc':now(),'before_any_fit':True,'sha256':hashes,'protocol_sha256':sha(E/'configs/protocol.json')})
 for p in files:p.chmod(0o444)
 (E/'configs/frozen.json').chmod(0o444)
 print('FROZEN',sha(E/'configs/frozen.json'),len(hashes),flush=True)
if __name__=='__main__':main()
