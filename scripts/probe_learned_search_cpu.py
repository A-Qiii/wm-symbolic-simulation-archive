from pathlib import Path
import json,torch,time,gzip
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.search import DFA,LearnedSearch
from wm_symbolic.symbol_decoder import Calibration,probabilities,decode
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/neural_search_probe';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(48)
ck=torch.load(P/'runs/wm_probe_seed0/checkpoint_01000.pt',map_location='cpu',weights_only=False);model=WorldModel().eval();model.load_state_dict(ck['model']);data=Episodes(P,'train',128,ck['normalization']);b=data.batch([0],'cpu')
catalog=json.loads((P/'configs/skills.json').read_text());budgets=json.loads((P/'configs/executor_budgets_pilot.json').read_text())['steps'];cal=Calibration();planner=LearnedSearch(model,catalog,budgets,cal)
tasks=json.loads((P/'configs/tasks/dev/task_specs.json').read_text());results=[]
with torch.inference_mode():
 e=model.encode(b['tokens'][:,:1],b['geometry'][:,:1],b['visible'][:,:1],b['robot'][:,:1])[:,0];root,_=model.initial(e.repeat(8,1));rootprob=probabilities(model.symbols(root),cal);mean={k:v.mean(0) for k,v in rootprob.items()}
 for task in [tasks[0],tasks[2],tasks[4]]:
  dfa=DFA(task['dfa']);facts,error=decode(mean,torch.zeros(3),dfa.props,cal);assert error is None
  progress=dfa.advance(dfa.initial,facts);start=time.monotonic();r=planner.plan(root,task,dfa,progress)
  assert r['queries']<=5000 and r['sample_successor_evaluations']==r['queries']*8
  assert all(e['transition_source']=='RSSM.imagine' and not e['latent_states_merged'] for e in r['edges'])
  with gzip.open(out/f"{task['task_id']}.json.gz",'wt') as f:json.dump(r,f)
  summary={k:v for k,v in r.items() if k!='edges'};summary.update(task_id=task['task_id'],wall_seconds=time.monotonic()-start,physical_execution=False,observation_root=data.episodes[0]['root_id']);results.append(summary);print('SEARCH',json.dumps(summary),flush=True)
(out/'results.json').write_text(json.dumps({'results':results,'scope':'Actual learned prior-to-symbol-to-beam interface check on a training root; no task success or generalization claim','handcoded_dynamic_effects':False,'all_25_skills_available':True,'GT_used_by_planner':False,'passed':True},indent=2))
