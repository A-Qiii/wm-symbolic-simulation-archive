from config import *
import sys,torch,fcntl,collections,time
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import Query,probs,feature_error

def main():
    start=time.monotonic();protocol=read(E/'configs/protocol.json')
    for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
    def guard(event,args):
        if event=='open'and isinstance(args[0],(str,bytes))and '/_private/'in str(args[0]):raise RuntimeError('Inference cannot open truth')
    sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
    data=dict(np.load(E/'data/public/inputs.npz'));z=torch.tensor(data['z'],device='cuda');o=torch.tensor(data['objects'],device='cuda');s=torch.tensor(data['scene'],device='cuda');ts=read(E/'data/public/trajectories.json');m,_,stats=load(0,'canonical_matched');frozen=tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})
    pack=dict(np.load(PACK/'inputs.npz'));pz=torch.tensor(pack['z'],device='cuda');po=torch.tensor(pack['objects'],device='cuda');ps=torch.tensor(pack['scene'],device='cuda');pairs=read(PACK/'pairs.json');pb=torch.tensor([r['before']for r in pairs],device='cuda');pa=torch.tensor([r['after']for r in pairs],device='cuda');aa=torch.tensor([r['action']for r in pairs],device='cuda');selected=[];curve=[];losses=[];qhashes={}
    def fh():return tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})
    def records(name,q,trajectories,u=None):
        out=[]
        for t in trajectories:
            rr=t['steps'];b=torch.tensor([r['before']for r in rr],device='cuda');a=torch.tensor([r['after']for r in rr],device='cuda');act=torch.tensor([r['action']for r in rr],device='cuda');one=m.transition(z[b],act);actual=z[a];paths={}
            cur=z[b[0]:b[0]+1];rec=[]
            for j in range(4):cur=m.transition(cur,act[j:j+1]);rec.append(cur[0])
            paths['full4']=(0,torch.stack(rec));paths['window0']=(0,torch.stack(rec[:3]));cur=z[b[1]:b[1]+1];rec=[]
            for j in range(1,4):cur=m.transition(cur,act[j:j+1]);rec.append(cur[0])
            paths['window1']=(1,torch.stack(rec))
            for path,(offset,pr)in paths.items():
                n=len(pr);before=b[offset:offset+n];after=a[offset:offset+n];values={'actual':actual[offset:offset+n],'one_step':one[offset:offset+n],'recursive':pr};pp={k:probs(q((v-stats['mean'])/stats['std'])).cpu().numpy()for k,v in values.items()};fe={k:feature_error(m.decode(v),o[after],s[after],(o[before],s[before])).cpu().numpy()for k,v in values.items()};lat={k:(v-z[after]).square().mean((1,2)).cpu().numpy()for k,v in values.items()}
                for j in range(n):out.append({k:t[k]for k in ['group','root_id','color','branch']}|{'condition':name,'checkpoint_update':u,'path':path,'global_step':offset+j+1,'local_depth':j+1,'skill':rr[offset+j]['skill'],'before':int(before[j]),'after':int(after[j]),**{k+'_probability':pp[k][j]for k in values},**{k+'_feature_groups':fe[k][j]for k in values},**{k+'_raw_MSE':float(lat[k][j])for k in values}})
        return out
    with torch.inference_mode():
        error=float((m.encode(o,s)-z).abs().max());assert error<1e-4,error
        for seed in [0,1]:
            q=Query().cuda();q.load_state_dict(torch.load(protocol['models']['query_heads'][str(seed)]['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);qhashes[seed]=tensor_hash(q.state_dict());sched=dict(np.load(STUDY/'configs'/f'schedule_seed{seed}.npz'));counts=np.bincount(sched['single'].ravel(),minlength=len(pairs));ix=np.flatnonzero(counts);weights=torch.tensor(counts[ix]/counts.sum(),device='cuda',dtype=torch.float32)
            for arm in ['old_chains','new_chains']:
                name=f'{arm}_seed{seed}';own='old_AB'if arm=='old_chains'else'new_AB';chosen=protocol['models']['selected'][name]['path'];m.load_state_dict(torch.load(chosen,weights_only=True,map_location='cuda')['state']);assert fh()==frozen;selected+=records(name,q,ts)
                chain=np.unique(sched[arm][3],axis=0);assert chain.shape==(24,3);chain=torch.tensor(chain,device='cuda')
                for u in [0,250,500,750,1000]:
                    cp=STUDY/'models'/name/f'update{u:04d}.pt';m.load_state_dict(torch.load(cp,weights_only=True,map_location='cuda')['state']);assert fh()==frozen;curve+=records(name,q,[t for t in ts if t['branch']=='normal'and t['group']in [own,'C_select','C_validate']],u)
                    singlef=[];singlez=[]
                    for k in range(0,len(ix),128):
                        ii=ix[k:k+128];pred=m.transition(pz[pb[ii]],aa[ii]);singlef.append(feature_error(m.decode(pred),po[pa[ii]],ps[pa[ii]],(po[pb[ii]],ps[pb[ii]])).mean(1));singlez.append((pred-pz[pa[ii]]).square().mean((1,2)))
                    sf=float((torch.cat(singlef)*weights).sum());sl=float((torch.cat(singlez)*weights).sum());cur=m.transition(pz[pb[chain[:,0]]],aa[chain[:,0]]);multi=[]
                    for depth in [1,2]:
                        ids=chain[:,depth];cur=m.transition(cur,aa[ids]);ff=float(feature_error(m.decode(cur),po[pa[ids]],ps[pa[ids]],(po[pb[ids]],ps[pb[ids]])).mean());ll=float((cur-pz[pa[ids]]).square().mean());multi.append({'depth':depth+1,'feature':ff,'latent_MSE':ll,'weighted_term':.25*(ff+.25*ll)})
                    losses.append({'condition':name,'update':u,'formally_selected':str(cp)==chosen,'single_feature':sf,'single_latent_MSE':sl,'single_loss':sf+.25*sl,'multi_terms':multi,'multi_update_total':sf+.25*sl+sum(x['weighted_term']for x in multi),'time_average_objective':sf+.25*sl+.25*sum(x['weighted_term']for x in multi),'definition':'Frozen checkpoint reevaluation with exact single-schedule frequencies and balanced24unique own training windows; not chronological SGD loss.'})
                    print('CHECKPOINT_DIAG',name,u,flush=True)
                assert tensor_hash(q.state_dict())==qhashes[seed]
    save(E/'results/predictions_selected.json',selected);save(E/'results/predictions_checkpoints.json',curve);save(E/'results/frozen_objective_curves.json',losses)
    save(E/'results/inference_audit.json',{'new_training_steps':0,'simulation_calls':0,'optimizer_created':False,'private_truth_reads':0,'selected_models_changed':False,'thresholds_changed':False,'encoder_decoder_unchanged':True,'query_hashes':qhashes,'reencoded_cache_max_abs_error':error,'selected_prediction_rows':len(selected),'diagnostic_checkpoint_rows':len(curve),'checkpoint_count':20,'observational_history_used':False,'seconds':time.monotonic()-start});print('FROZEN_DIAG_DONE',len(selected),len(curve),flush=True)
if __name__=='__main__':main()
