from config import *
import ast,sys,torch
from train import loss_function
sys.path.insert(0,str(REP/'source'))
from model import Query
def main():
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 cfg=read(E/'configs/protocol.json');q=Query();q.load_state_dict(torch.load(cfg['initial_heads']['0']['path'],weights_only=True,map_location='cpu')['state']);loss=loss_function();counts={}
 for arm,n in [('control',156),('coverage',162)]:
  d=np.load(E/'data/train'/f'{arm}.npz');assert d['z'].shape==(n,4,64)and d['y'].shape==d['v'].shape==(n,9);assert np.isfinite(d['z']).all();z,y,v=[torch.tensor(d[k][:32])for k in ['z','y','v']];out=q(z);ls=loss(out,y,v,torch.tensor(cfg['class_weights']));assert out.shape==(32,9)and ls.shape==(32,)and torch.isfinite(ls).all();counts[arm]={'states':n,'roots':len(set(d['root_ids'])),'z_dtype':str(d['z'].dtype),'y_dtype':str(d['y'].dtype),'v_dtype':str(d['v'].dtype)}
 ev={}
 for f in (E/'data/evaluation').glob('*.npz'):
  d=np.load(f);meta=read(f.with_name(f.stem+'_meta.json'));assert len(meta)==len(d['z'])==len(d['y']);assert d['z'].shape[1:]==(4,64)and d['y'].shape[1:]==(9,);assert np.isfinite(d['z']).all();ev[f.stem]={'states':len(d['z']),'roots':len(set(d['root_ids'])),'full9_valid':int(d['v'].all(1).sum())}
 assert sum(v['states']for v in ev.values())==1089
 p=np.array([[.5,.49,.51,.1,.9,.2,.2,.7,.1]],np.float32);assert discrete(p).tolist()==[[True,False,True,False,True,False,False,True,False]]
 save(E/'audit/preflight.json',{'utc':now(),'status':'passed','syntax_all_sources':True,'original_Query_forward_and_loss_finite':True,'optimizer_created':False,'training':counts,'evaluation':ev,'prepare_process_final_stdout_error':'Initial prepare completed all data/config saves; final json.dumps of numpy arrays in status print raised TypeError. Print-only serialization fixed, preparation not rerun. All artifacts independently validated here.'});print('QUERY_CONTROL_PREFLIGHT',counts,ev,flush=True)
if __name__=='__main__':main()
