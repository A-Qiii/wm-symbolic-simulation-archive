from common import *
from model import QueryRelationReader,parse_query,decode
from train import load_inputs
import argparse,torch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['pooled','spatial'],default='spatial');ap.add_argument('--window-id',required=True);ap.add_argument('--query',default='Closed(drawer)');ap.add_argument('--goal-object',choices=COLORS,default='red');a=ap.parse_args();torch.set_num_threads(4);meta=read(BASE/'data/metadata.json');idx=next(i for i,r in enumerate(meta)if r['window_id']==a.window_id);raw=load_inputs();x={k:torch.tensor(v[idx:idx+1])for k,v in raw.items()};cp=read(E/'configs'/a.variant/'reader_frozen.json');assert sha(cp['path'])==cp['sha256'];net=QueryRelationReader(a.variant).eval();net.load_state_dict(torch.load(cp['path'],weights_only=False,map_location='cpu')['model']);q=parse_query(a.query,{'obj_'+c:j for j,c in enumerate(COLORS)})
 with torch.inference_mode():nodes=net.nodes(**x);answer=float(net.query(nodes,torch.tensor([q]))[0,0]);prob=net.probabilities(**x).numpy();d=decode(prob)[0]
 target=COLORS.index(a.goal_object);print(json.dumps({'variant':a.variant,'query':a.query,'probability':answer,'all_probabilities':dict(zip(FACTS,map(float,prob[0]))),'decoded_state':dict(zip(FACTS,map(int,d))),'joint_goal':f'Closed(drawer) AND In(obj_{a.goal_object},drawer)','joint_goal_currently_satisfied':bool(d[6]==1 and d[3+target]==1),'GT_loaded_at_inference':False,'planning_status':'not_executed; prior learned operator support flags unchanged','scope':'Saved public-window development inference; not a new physical planning run'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
