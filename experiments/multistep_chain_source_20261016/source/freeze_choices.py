from config import *
def main():
    assert not(E/'configs/choices_frozen.json').exists()
    for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
    selected={};initials=[];paths=[]
    for seed in [0,1]:
        for arm in ['old_chains','new_chains']:
            name=f'{arm}_seed{seed}';s=read(E/'models'/name/'selected.json');t=read(E/'results'/f'train_{name}.json');assert t['updates']==1000 and t['single_draws']==64000 and t['multistep_draws']==12000
            assert s['C_score']==min(x['score']for x in t['log']);assert sha(s['path'])==s['sha256'];assert t['frozen_encoder_decoder_hash']==t['final_frozen_hash']and t['query_hash']==t['final_query_hash'];selected[name]=s;initials.append(t['initial_model_hash']);paths+=[Path(s['path']),E/'results'/f'train_{name}.json']
    assert len(set(initials))==1
    save(E/'configs/choices_frozen.json',{'frozen_utc':now(),'selected':selected,'query_heads':read(E/'configs/protocol.json')['query_heads'],'criterion':'unchanged old/new C-select equal root-macro one-step feature error; earlier checkpoint','C_validation_or_test_used':False,'sha256':{str(p):sha(p)for p in paths}})
    print('CHOICES_FROZEN before C_backtest and conditional D5',flush=True)
if __name__=='__main__':main()
