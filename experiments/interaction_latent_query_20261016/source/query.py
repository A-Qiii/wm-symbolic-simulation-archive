"""Runnable public observation query and learned successor API. No execution."""
from common import *
from worker_fixed_batch import Engine
from model import probs
from train_queries import normalize
from representation import transform
import argparse,fcntl,torch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--observation',required=True);ap.add_argument('--arm',choices=['generic','ae','interaction'],default='interaction');ap.add_argument('--target',choices=COLORS,required=True);ap.add_argument('--skill',choices=['open_drawer','close_drawer','place_in_drawer']+[f'lift_{c}_block_table'for c in COLORS]);ap.add_argument('--argument',choices=COLORS);ap.add_argument('--query');ap.add_argument('--output');a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);engine=Engine();result=engine.handle({'kind':a.arm,'public_path':a.observation,'target':COLORS.index(a.target)});result['named_current_queries']=dict(zip(FACTS,result.pop('current_probability')));result['physical_actions_executed']=0;result['model_seed']=0
 if a.query:
  query=a.query.replace('obj_','').replace(' ','');assert query in FACTS,f'Allowed shared queries: {FACTS}';result['requested_query']={'query':query,'probability':result['named_current_queries'][query]}
 if a.skill:
  ff=engine.encoder.encode([Path(a.observation)]*8)[0];o,s=transform({k:v[None]for k,v in ff.items()},engine.norm);m,q,stats=engine.models[a.arm];argument=COLORS.index(a.argument)if a.argument else COLORS.index(a.target)if a.skill=='place_in_drawer'else None;act=action(a.skill,argument,96)
  with torch.no_grad():z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda'));nextz=m.transition(z,torch.tensor([act],device='cuda'));prob=probs(q(normalize(nextz,stats)))[0].cpu().tolist()
  result['declared_skill']=a.skill;result['declared_action_vector']=act;result['named_predicted_successor_queries']=dict(zip(FACTS,prob));result['successor_source']='learned continuous transition, no hand-coded effect'
 if a.output:save(a.output,result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
