from common import *
from model import QueryRelationReader,parse_query,decode
from learn_actions import search
import torch,argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--window-id',required=True);ap.add_argument('--query',default='Closed(drawer)');ap.add_argument('--goal-object',choices=COLORS,default='red');a=ap.parse_args();torch.set_num_threads(4);meta=read(E/'data/metadata.json');idx=next(i for i,r in enumerate(meta)if r['window_id']==a.window_id);raw=dict(np.load(E/'data/inputs.npz'));x={k:torch.tensor(v[idx:idx+1])for k,v in raw.items()};cp=read(E/'configs/reader_frozen.json');assert sha(cp['path'])==cp['sha256'];net=QueryRelationReader().eval();net.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=False)['model']);q=parse_query(a.query,{'obj_'+c:j for j,c in enumerate(COLORS)})
 with torch.inference_mode():nodes=net.nodes(**x);answer=float(net.query(nodes,torch.tensor([q]))[0,0]);prob=net.probabilities(**x).numpy();state=decode(prob)[0]
 target=COLORS.index(a.goal_object);result={'query':a.query,'probability':answer,'decoded_current_state':dict(zip(FACTS,state.tolist())),'goal_text':f'In(obj_{a.goal_object},drawer) AND Closed(drawer)','goal_currently_satisfied':bool(state[3+target]==1 and state[6]==1),'learned_rule_search':search(state,[(3+target,1),(6,1)],read(E/'configs/learned_rules9.json')['operators']),'GT_read_at_inference':False,'warning':'AB/C saved public-window inference, not a new closed-loop execution. Unsupported learned rules cannot be used as deterministic transitions.'};print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
