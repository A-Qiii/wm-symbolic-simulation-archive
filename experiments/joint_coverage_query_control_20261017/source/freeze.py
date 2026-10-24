from config import *
import ast,shutil

def main():
 assert not(E/'configs/frozen.json').exists();cfg=read(E/'configs/protocol.json');source=read(E/'configs/source_references.json');refs=dict(source['sha256'])
 for p,h in refs.items():assert sha(p)==h,p
 tr={a:dict(np.load(E/'data/train'/f'{a}.npz'))for a in ['control','coverage']}
 for k in ['z','y','v','root_ids','state_ids']:assert np.array_equal(tr['control'][k],tr['coverage'][k][:156]),k
 forbidden=set(cfg['root_split']['validation'])|set(cfg['root_split']['diagnostic_D_seen']);assert all(set(d['root_ids']).isdisjoint(forbidden)for d in tr.values());assert set(tr['coverage']['root_ids'][156:])==set(cfg['root_split']['train'])
 for seed in [0,1]:
  s=np.load(E/'configs'/f'schedule_seed{seed}.npz');assert s['control'].shape==s['coverage'].shape==(600,32);assert np.array_equal(s['control'][:,:24],s['coverage'][:,:24]);assert np.all(s['coverage'][:,24:]>=156)and np.all(s['coverage'][:,24:]<162)and np.all(s['control']<156)
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
 for p in E.rglob('*'):
  if p.is_file()and '__pycache__'not in p.parts:refs[str(p)]=sha(p)
 save(E/'configs/frozen.json',{'utc':now(),'optimizer_updates_started':False,'sha256':refs,'prior_manifests':source['prior_manifests'],'allowed_fits':4,'updates_each':600,'threshold':.5,'same_architecture_and_initialization_paired':True});print('QUERY_CONTROL_FROZEN',len(refs),'refs;4fits;600steps;25percentnewcoverage',flush=True)
if __name__=='__main__':main()
