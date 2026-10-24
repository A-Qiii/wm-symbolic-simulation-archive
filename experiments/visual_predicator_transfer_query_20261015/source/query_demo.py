"""Executable template query and learned-rule planning over a public feature window."""
from common import *
from model import QueryRelationReader,parse_query
from symbolic import search
import argparse,torch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True,help='public normalized objects/context/robot NPZ');ap.add_argument('--query',default='Holding(obj_red)');ap.add_argument('--goal-object',choices=COLORS,default='red');a=ap.parse_args();torch.set_num_threads(4);cp=read(E/'configs/reader_frozen.json');assert sha(cp['path'])==cp['sha256'];model=QueryRelationReader().eval();model.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=False)['model']);raw=dict(np.load(a.inputs));assert set(raw)=={'objects','context','robot'};x={k:torch.tensor(v[None]if v.ndim=={'objects':3,'context':2,'robot':2}[k]else v)for k,v in raw.items()};entities={'obj_'+c:i for i,c in enumerate(COLORS)};q=parse_query(a.query,entities)
 with torch.inference_mode():nodes=model.nodes(**x);answer=model.query(nodes,torch.tensor([q])).sigmoid().numpy();pro=model(**x).sigmoid().numpy()
 goal=[(3+COLORS.index(a.goal_object),1),(6,0)];result={'query':a.query,'query_probability':float(answer[0,0]),'raw_query_value':bool(answer[0,0]>=.5),'all7_consistency_checked_state':dict(zip(FACTS,decode(pro)[0].tolist())),'entity_types':{'block':list(entities),'drawer':['drawer'],'gripper':['gripper']},'goal':goal,'learned_plan':search(decode(pro)[0],goal,read(E/'configs/rules_frozen.json')['operators']),'caveat':'Template vocabulary only; raw query probability must retain global multiple-Holding UNKNOWN protection when consumed by planner.'};print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
