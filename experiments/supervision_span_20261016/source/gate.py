from config import *
def main():
    assert not(E/'results/gate.json').exists();m=read(E/'results/metrics.json');cum=read(E/'results/cumulative.json');f=read(E/'configs/choices_frozen.json');results=[]
    def get(name,g):return sorted([r for r in m if r['condition']==name and r['group']==g and r['branch']=='normal'and r['path']=='full4'and r['color']=='all'],key=lambda r:r['global_step'])
    def cumulative(name,g):return next(r['all4_correct']for r in cum if r['condition']==name and r['group']==g and r['branch']=='normal'and r['color']=='all'and r['method']=='recursive')
    for seed in[0,1]:
        a=f'span3_seed{seed}';b=f'span4_seed{seed}';old=get(a,'C_validate');new=get(b,'C_validate');checks={'C_multistep_feature_better':f['selected'][b]['C_score']<f['selected'][a]['C_score'],'C_step3_no_loss':new[2]['recursive']['full9_correct']>=old[2]['recursive']['full9_correct'],'C_step4_gain_at_least1':new[3]['recursive']['full9_correct']>=old[3]['recursive']['full9_correct']+1,'C_late_sum_gain_at_least2':sum(r['recursive']['full9_correct']for r in new[2:])>=sum(r['recursive']['full9_correct']for r in old[2:])+2,'C_cumulative_gain_at_least1':cumulative(b,'C_validate')>=cumulative(a,'C_validate')+1,'AB_cumulative_loss_at_most1':cumulative(b,'new_AB')>=cumulative(a,'new_AB')-1}
        for j in[0,1]:checks[f'C_early_h{j+1}_loss_at_most1']=new[j]['recursive']['full9_correct']>=old[j]['recursive']['full9_correct']-1
        for j in range(4):checks[f'C_one_step_h{j+1}_loss_at_most1']=new[j]['one_step']['full9_correct']>=old[j]['one_step']['full9_correct']-1
        results.append({'seed':seed,'checks':checks,'passed':all(checks.values())})
    passed=all(r['passed']for r in results);save(E/'results/gate.json',{'evaluated_utc':now(),'protocol_sha256':sha(E/'configs/protocol.json'),'passed':passed,'seeds':results,'action':'Freeze newD5 roots and models then collect one fixed test'if passed else'Stop, no newD collection, no further training/tuning'});print('GATE',passed,json.dumps(results),flush=True)
if __name__=='__main__':main()
