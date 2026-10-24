from common_diag import *
import torch,fcntl
from torch.nn import functional as F

def main():
 q,d,z,t=data();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 reader,logits,digest,balanced,Net=old_modules();m=reader().double();net=Net('full').cuda().double();net.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=False)['model']);params=list(net.parameters());rdigest=digest(m)
 d={k:v.double()if v.dtype!=torch.bool else v for k,v in d.items()};w=torch.tensor(q['class_weights'],device='cuda',dtype=torch.float64);y=d['after_y'];v=d['after_valid'];N=137
 meta=read(PREV/'data/training/_private/metadata.json');pub=read(PREV/'data/training/public/pairs.json');priv={r['pair_id']:r for r in read(B/'data_manifest.json')['private_pairs']};rows=[]
 schedules={s:np.load(PREV/'configs'/f'shared_batch_schedule_seed{s}.npz')['local_indices']for s in [0,1,2]}
 with torch.no_grad():
  cur=logits(m,d['current_objects'],d['current_scene']).sigmoid().cpu().numpy();nxt=logits(m,d['next_objects'],d['next_scene']).sigmoid().cpu().numpy();h,s=net(d['current_objects'],d['current_scene'],d['declared_actions']);pred=logits(m,h,s).sigmoid().cpu().numpy()
 for local,i in enumerate(q['joint_global_ids']):
  p=pub[i];a=np.zeros(9,np.float32);kind=next(k for k in ['open','lift','place','close']if p['skill'].startswith(k));a[['open','lift','place','close'].index(kind)]=1
  if kind=='lift':a[4+COLORS.index(p['declared_arguments']['color'])]=1
  a[7:]=[p['declared_primitive_budget']/900,p['declared_posthold_controls']/96];assert np.array_equal(a,z['declared_actions'][i]),p
  if local<132:continue
  pr=priv[p['pair_id']];r=meta[i];target=COLORS.index(p['declared_arguments']['color']);held=COLORS.index(r['other_colour_evaluation_only'])
  assert target!=held and pr['primitive']['status']=='refused'and pr['primitive']['reason']=='requires_empty_gripper'
  for side in ['before','after']:
   assert np.array_equal(t[side+'_y'][i],pr[side+'_targets']['y']);assert np.array_equal(t[side+'_valid'][i],pr[side+'_targets']['valid']);assert t[side+'_valid'][i].all()
  expected=np.zeros(7);expected[held]=1;expected[6]=1;assert np.array_equal(t['after_y'][i],expected)and np.array_equal(t['before_y'][i],expected)
  pre=read(r['preaction_prediction_record_path']);post=read(r['after_observation_record_path']);bo=np.load(pre['observed_path']);ao=np.load(post['observed_path']);ap=np.load(pre['predictions_path'])
  assert np.array_equal(z['current_objects'][i],bo['current_objects'][0])and np.array_equal(z['current_scene'][i],bo['current_scene'][0]);assert np.array_equal(t['next_objects'][i],ao['current_objects'][0])and np.array_equal(t['next_scene'][i],ao['current_scene'][0]);assert np.array_equal(z['declared_actions'][i],ap['declared_actions'][0])
  counts={str(seed):int((a==local).sum())for seed,a in schedules.items()};assert set(counts.values())=={175}
  rows.append({'global_index':i,'local_index':local,'pair_id':p['pair_id'],'target':COLORS[target],'held_other':COLORS[held],'action':a.tolist(),'y_before':t['before_y'][i].tolist(),'y_after':t['after_y'][i].tolist(),'valid':t['after_valid'][i].tolist(),'actual_outcome':'refused_requires_empty_gripper','appearances_each_prior_seed':counts,'raw_source_values_identical':True,'current_probabilities':cur[local].tolist(),'next_observed_probabilities':nxt[local].tolist(),'failed_model_probabilities':pred[local].tolist(),'current_all7':correctness(cur[local:local+1],t['before_y'][i:i+1])['all7'],'next_all7':correctness(nxt[local:local+1],t['after_y'][i:i+1])['all7']})
 assert all(r['current_all7']==r['next_all7']==1 for r in rows)
 save(E/'results/sample_audit.json',{'passed':True,'actions_checked':137,'rows':rows,'all_five_valid_35of35':True,'object_slots':['red','blue','pink'],'shared_head_output_order':NAMES,'current_real_latent_readout':5,'next_real_latent_readout':5,'cached_current_max_probability_difference':float(np.max(np.abs(cur-z['observed_current_probabilities'][q['joint_global_ids']]))),'cached_next_max_probability_difference':float(np.max(np.abs(nxt-t['observed_next_probabilities'][q['joint_global_ids']]))),'label_source':str(B/'data_manifest.json')})

 def outputs():
  hh,ss=net(d['current_objects'],d['current_scene'],d['declared_actions']);return hh,ss,logits(m,hh,ss)
 def components(ids):
  hh,ss,ll=outputs();ll=ll[ids];yy=y[ids];vv=v[ids];raw=F.softplus(ll)-yy*ll;parts={}
  for gi,(name,cols)in enumerate([('held',[0,1,2]),('in',[3,4,5]),('open',[6])]):
   ww=torch.where(yy[:,cols]>.5,w[gi,1],w[gi,0]);parts[name]=(raw[:,cols]*ww*vv[:,cols]).sum()/vv[:,cols].sum()/3
  parts['object_latent']=.025*F.smooth_l1_loss(hh[ids],d['next_objects'][ids]);parts['scene_latent']=.025*F.smooth_l1_loss(ss[ids],d['next_scene'][ids]);parts['BCE']=sum(parts[k]for k in ['held','in','open']);parts['latent']=parts['object_latent']+parts['scene_latent'];parts['total']=parts['BCE']+parts['latent'];assert abs(float(parts['BCE']-balanced(ll,yy,vv,w)))<1e-10
  return parts
 def grad(loss):
  gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);return torch.cat([(torch.zeros_like(p)if g is None else g).reshape(-1)for p,g in zip(params,gg)])
 groups={'old132':slice(0,132),'five':slice(132,137),'joint137':slice(None)};g={};losses={}
 for name,ids in groups.items():
  comp=components(ids);g[name]={k:grad(x).detach()for k,x in comp.items()};losses[name]={k:{'loss':float(x.detach()),'gradient_L2':float(g[name][k].norm())}for k,x in comp.items()}
 hh,ss,ll=outputs();newtargets=torch.tensor([COLORS.index(meta[i]['planned_colour_evaluation_only'])for i in q['five_global_ids']],device='cuda');li=torch.arange(132,137,device='cuda');target_logits=ll[li,newtargets];target_y=y[li,newtargets];assert(target_y==0).all();target_losses=F.binary_cross_entropy_with_logits(target_logits,target_y,reduction='none')*w[0,0]/45
 gt=grad(target_losses.sum()).detach();per=[]
 for j in range(5):
  l=target_losses[j];gl=torch.autograd.grad(l,target_logits,retain_graph=True)[0][j];gh=torch.autograd.grad(l,hh,retain_graph=True)[0][132+j,newtargets[j]];gp=grad(l);eps=1e-4;zj=target_logits[j].detach();finite=(F.softplus(zj+eps)-F.softplus(zj-eps))*w[0,0]/45/(2*eps)
  per.append({'pair_id':rows[j]['pair_id'],'target_probability':float(zj.sigmoid()),'raw_target_BCE':float(F.softplus(zj)),'contribution_to_five_total':float(l.detach()),'d_loss_d_target_logit':float(gl),'analytic_derivative':float(zj.sigmoid()*w[0,0]/45),'finite_difference':float(finite),'target_latent_gradient_L2':float(gh.norm()),'parameter_gradient_L2':float(gp.norm()),'valid':True})
  assert gl>0 and gh.norm()>0 and gp.norm()>0 and abs(float(gl-finite))<1e-8
 def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp(min=1e-30))
 pairs=[]
 for a,ka,b,kb in [('five','BCE','old132','BCE'),('five','BCE','five','latent'),('five','total','old132','total'),('joint137','BCE','joint137','latent')]:pairs.append({'a':a+'.'+ka,'b':b+'.'+kb,'cosine':cos(g[a][ka],g[b][kb])})
 influences=[]
 for name,vec in [('five_total',g['five']['total']),('old132_total',g['old132']['total']),('uniform_joint',g['joint137']['total']),('equal_group_joint',.5*g['five']['total']+.5*g['old132']['total'])]:influences.append({'direction':'negative_gradient_'+name,'target_BCE_first_order_change_per_unit_step':float(-torch.dot(gt,vec)),'cosine_with_target_BCE_gradient':cos(gt,vec),'gradient_L2':float(vec.norm())})
 # A two-sided parameter directional finite difference, restored exactly; no optimizer.
 originals=[p.detach().clone()for p in params];direction=gt/gt.norm();eps=1e-4;vals=[]
 for sign in [1,-1]:
  offset=0
  with torch.no_grad():
   for p,b in zip(params,originals):p.copy_(b+sign*eps*direction[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
   _,_,ll=outputs();vals.append(float(F.softplus(ll[li,newtargets]).sum()*w[0,0]/45))
 with torch.no_grad():
  for p,b in zip(params,originals):p.copy_(b)
 fd=(vals[0]-vals[1])/(2*eps);assert abs(fd-float(gt.norm()))/max(float(gt.norm()),1e-15)<1e-4
 assert digest(m)==rdigest and all(p.grad is None for p in m.parameters())
 save(E/'results/gradient_audit.json',{'precision':'float64 for numerical diagnosis only; diagnostic training retains historicalfloat32 high matmul','no_optimizer_updates':True,'loss_manual_matches_function':True,'group_components':losses,'five_target_facts':per,'cosines':pairs,'target_loss_directions':influences,'parameter_finite_difference':{'expected':float(gt.norm()),'observed':fd,'epsilon':eps},'frozen_reader_no_parameter_grad':True,'five_mask_valid':35,'new_group_weight_uniform':5/137,'new_group_weight_equal':.5,'new_per_case_weight_ratio_equal_to_uniform':13.7,'gradient_sum_relative_error':float((g['joint137']['total']-(132*g['old132']['total']+5*g['five']['total'])/137).norm()/g['joint137']['total'].norm())})

 near=[]
 with torch.no_grad():
  for j,i in enumerate(q['five_global_ids']):
   n=132+j;target=int(newtargets[j]);a=d['declared_actions'];same=(a[:132]==a[n]).all(1);success=(y[:132,target]==1);ids=torch.where(same&success)[0];assert len(ids)>0
   def objinput(k):return torch.cat([d['current_objects'][k,target],d['current_scene'][k],a[k,:4],a[k,4+target:5+target],a[k,7:9]])
   xx=objinput(n);dist=torch.stack([(objinput(int(k))-xx).square().mean().sqrt()for k in ids]);k=int(ids[dist.argmin()]);other=COLORS.index(rows[j]['held_other']);hh,ss=net(d['current_objects'][n:n+1],d['current_scene'][n:n+1],a[n:n+1]);modified=d['current_objects'][n:n+1].clone();modified[0,other]=d['current_objects'][k,other];h2,s2=net(modified,d['current_scene'][n:n+1],a[n:n+1]);ownerr=float((hh[0,target]-h2[0,target]).abs().max());assert ownerr==0
   near.append({'new_pair':rows[j]['pair_id'],'nearest_success_pair':pub[q['joint_global_ids'][k]]['pair_id'],'candidate_count_same_declared_action':len(ids),'target_input199_RMS':float(dist.min()),'target_h96_RMS':float((d['current_objects'][n,target]-d['current_objects'][k,target]).square().mean().sqrt()),'global_s96_RMS':float((d['current_scene'][n]-d['current_scene'][k]).square().mean().sqrt()),'held_other_h96_RMS':float((d['current_objects'][n,other]-d['current_objects'][k,other]).square().mean().sqrt()),'exact_target_input_collision':bool(torch.equal(objinput(n),objinput(k))),'other_object_only_replacement_target_output_max_delta':ownerr,'nearest_current_probs':cur[k].tolist(),'new_current_probs':cur[n].tolist()})
 save(E/'results/input_information_audit.json',{'rows':near,'frozen_reader_reads_held_other_in_all5':True,'target_update_no_direct_other_object_input':True,'global_context_may_contain_other_held_information':True,'no_architectural_impossibility_claim':True,'no_new_encoding_or_simulation':True})
 save(E/'results/audit_complete.json',{'created_utc':now(),'passed':True,'optimizer_updates':0,'C_D_model_forwards':0,'training_can_start':True})
 print(json.dumps({'audit_passed':True,'samples':rows,'gradients':losses,'cosines':pairs,'directions':influences,'nearest':near},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
