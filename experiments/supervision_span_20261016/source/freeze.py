from config import *
def main():
    assert not(E/'configs/choices_frozen.json').exists()
    for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
    paths=[];selected={};initials=[]
    for seed in[0,1]:
        for arm in['span3','span4']:
            name=f'{arm}_seed{seed}';r=read(E/'results'/f'train_{name}.json');s=read(E/'models'/name/'selected.json');assert r['updates']==1000 and r['single_draws']==64000 and r['multi_starts']==12000;best=min(r['log'],key=lambda x:x['score']);assert s['path'].endswith(f'update{best["update"]:04d}.pt')and s['C_score']==best['score'];assert r['frozen_encoder_decoder_hash']==r['final_frozen_hash']and r['query_hash']==r['final_query_hash'];selected[name]=s;initials.append(r['initial_model_hash']);paths+=[Path(s['path']),E/'results'/f'train_{name}.json']
    assert len(set(initials))==1;save(E/'configs/choices_frozen.json',{'frozen_utc':now(),'selected':selected,'query_heads':read(E/'configs/protocol.json')['query_heads'],'C_validation_or_test_used':False,'sha256':{str(p):sha(p)for p in paths}});print('CHOICES_FROZEN',flush=True)
if __name__=='__main__':main()
