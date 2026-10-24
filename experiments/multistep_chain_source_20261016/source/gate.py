from config import *
def main():
    assert not(E/'results/gate.json').exists();m=read(E/'results/metrics_C_backtest.json');f=read(E/'configs/choices_frozen.json');results=[]
    for seed in [0,1]:
        new=m['trajectory_prediction'][f'new_chains_seed{seed}']['normal'];checks={'C_selection':f['selected'][f'new_chains_seed{seed}']['C_score']<f['selected'][f'old_chains_seed{seed}']['C_score']}
        for arm in ['initial','old_chains']:
            ref=m['trajectory_prediction'][f'{arm}_seed{seed}']['normal'];checks[arm+'_step3_gain']=new[2]['recursive']['full9_correct']>=ref[2]['recursive']['full9_correct']+1
            checks[arm+'_step3plus4_gain']=sum(x['recursive']['full9_correct']for x in new[2:])>=sum(x['recursive']['full9_correct']for x in ref[2:])+2
            for step,method in [(0,'recursive'),(1,'recursive'),(2,'one_step'),(3,'one_step')]:checks[f'{arm}_{method}_h{step+1}_retention']=new[step][method]['full9_correct']>=ref[step][method]['full9_correct']-1
        results.append({'seed':seed,'checks':checks,'passed':all(checks.values())})
    passed=all(r['passed']for r in results)
    save(E/'results/gate.json',{'evaluated_utc':now(),'protocol_sha256':sha(E/'configs/protocol.json'),'passed':passed,'seeds':results,'action':'Freeze and collect one12-rootD5 batch'if passed else'No D5 collection, no additional tuning; report finite negative/inconclusive result','not_statistical_significance':True})
    print('GATE',passed,json.dumps(results),flush=True)
if __name__=='__main__':main()
