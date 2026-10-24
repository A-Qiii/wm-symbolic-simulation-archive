from config import *
def main():
 assert not(E/'results/gate.json').exists();m=read(E/'results/metrics.json');c=read(E/'results/cumulative.json');p=read(E/'results/predictions.json');results=[]
 def get(n,g):return sorted([r for r in m if r['condition']==n and r['group']==g and r['branch']=='normal'and r['path']=='full4'and r['color']=='all'],key=lambda r:r['global_step'])
 def cumulative(n,g):return next(r['all4_correct']for r in c if r['condition']==n and r['group']==g and r['branch']=='normal'and r['color']=='all'and r['method']=='recursive')
 lookup={(r['condition'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in p}
 for seed in [0,1]:
  a=f'mixed_seed{seed}';b=f'latent_seed{seed}';old=get(a,'C_validate');new=get(b,'C_validate');checks={f'C_step{j+1}_gain_at_least1':new[j]['recursive']['full9_correct']>=old[j]['recursive']['full9_correct']+1 for j in [2,3]};checks['C_all4_gain_at_least1']=cumulative(b,'C_validate')>=cumulative(a,'C_validate')+1
  for j in [0,1]:checks[f'C_early_h{j+1}_loss_at_most1']=new[j]['recursive']['full9_correct']>=old[j]['recursive']['full9_correct']-1
  for j in range(4):checks[f'C_one_step_h{j+1}_loss_at_most1']=new[j]['one_step']['full9_correct']>=old[j]['one_step']['full9_correct']-1
  for g in ['old_AB','new_AB']:checks[g+'_all4_loss_at_most1']=cumulative(b,g)>=cumulative(a,g)-1
  checks['actual_readout_identical']=all(r['actual_probability']==lookup[(b,r['root_id'],r['branch'],r['path'],r['global_step'])]['actual_probability']for r in p if r['condition']==a)
  results.append({'seed':seed,'checks':checks,'passed':all(checks.values())})
 passed=all(r['passed']for r in results);save(E/'results/gate.json',{'evaluated_utc':now(),'protocol_sha256':sha(E/'configs/protocol.json'),'passed':passed,'seeds':results,'action':'Report to parent for review; no D collection or adoption authorized'if passed else'Stop; no further tuning or D collection','newD_collected':False,'formal_adoption':False});print('GATE',passed,json.dumps(results),flush=True)
if __name__=='__main__':main()
