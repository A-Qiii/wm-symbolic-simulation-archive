from config import *
import csv,collections

def csvout(path,rows):
 with path.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def semantics(logit,y):
 pred=discrete_logits(logit);ok=pred==y;bm=np.where(y[...,:6],logit[...,:6],-logit[...,:6]);cl=y[...,6:].argmax(-1);v=logit[...,6:];true=np.take_along_axis(v,cl[...,None],-1)[...,0];comp=v.copy();np.put_along_axis(comp,cl[...,None],-np.inf,-1)
 return ok.all(-1),np.stack([ok[...,:3].all(-1),ok[...,3:6].all(-1),ok[...,6:].all(-1)],-1),np.stack([bm[...,:3].min(-1),bm[...,3:6].min(-1),true-comp.max(-1)],-1)
def main():
 samples=read(E/'configs/samples.json');cases=read(E/'data/cases.json');labs=read(DIAG/'data/_private/labels.json');d=dict(np.load(E/'data/latent_probes.npz'));h=d['history'];cp=dict(np.load(E/'data/checkpoints.npz'))['values'];cpm=read(E/'data/checkpoint_cases.json');ys=np.array([labs[str(s['after'])]['y']for s in samples],bool);valid=np.array([labs[str(s['after'])]['v']for s in samples]);assert valid.all();y=ys[[c['sample_id']for c in cases]]
 ok,gok,margin=semantics(h[...,5:14],np.broadcast_to(y[:,None,None],h[...,5:14].shape));F=h[...,:4].mean(-1);Z=h[...,4];mixed=F+.25*Z;objectives=np.stack([mixed[:,0],.25*Z[:,1]],1);f0=F[:,0,0];z0=Z[:,0,0];ok0=ok[:,0,0];gok0=gok[:,0,0];margin0=margin[:,0,0];pred0=discrete_logits(h[:,0,0,5:14]);steps=read(E/'configs/protocol.json')['optimization']['snapshots'];eps=1e-6
 tags=[]
 for i,c in enumerate(cases):
  s=samples[c['sample_id']];tag={k:s[k]for k in ['split','group','root_id','color','branch','global_step','skill']};tag.update({'truth_Holding':'held'if y[i,:3].any()else'empty','truth_In':'inside'if y[i,3:6].any()else'outside','truth_drawer':['Closed','Open','Intermediate'][y[i,6:].argmax()]});tags.append(tag)
 cohorts={'all':np.ones(len(cases),bool),'baseline_correct':ok0,'baseline_wrong':~ok0}
 for g,sl in [('Holding',slice(0,3)),('In',slice(3,6))]:
  cohorts[g+'_false_positive']=((pred0[:,sl]&~y[:,sl]).any(1));cohorts[g+'_false_negative']=((~pred0[:,sl]&y[:,sl]).any(1))
 cohorts['drawer_error']=~gok0[:,2]
 for field in ['truth_Holding','truth_In','truth_drawer','global_step','color']:
  for value in sorted({t[field]for t in tags}):cohorts[field+'='+str(value)]=np.array([t[field]==value for t in tags])
 summaries=[]
 for name,origin in sorted({(c['model'],c['origin'])for c in cases}):
  base=np.array([c['model']==name and c['origin']==origin for c in cases]);seed=next(c['seed']for c in cases if c['model']==name)
  for split in ['all','AB_train','C_development']:
   for branch in ['all','normal','held_closed']:
    use=base&np.array([split=='all'or t['split']==split for t in tags])&np.array([branch=='all'or t['branch']==branch for t in tags])
    for cohort,mask in cohorts.items():
     # Fine state/error cohorts are reported within each split, branch pooled; main table also separates branches.
     if branch!='all'and cohort!='all':continue
     ii=np.where(use&mask)[0]
     if not len(ii):continue
     for oi,obj in enumerate(['mixed','latent_only']):
      for si,step in enumerate(steps):
       good=ok[ii,oi,si];lossdown=objectives[ii,oi,si]<objectives[ii,oi,0]-eps;bothdown=(F[ii,oi,si]<f0[ii]-eps)&(Z[ii,oi,si]<z0[ii]-eps)
       r={'model':name,'seed':seed,'origin':origin,'objective':obj,'step':step,'split':split,'branch':branch,'cohort':cohort,'N_boundaries':len(ii),'N_roots':len({tags[i]['root_id']for i in ii}),'baseline_correct':int(ok0[ii].sum()),'correct':int(good.sum()),'correct_to_wrong':int((ok0[ii]&~good).sum()),'wrong_to_correct':int((~ok0[ii]&good).sum()),'objective_down':int(lossdown.sum()),'objective_down_correct_to_wrong':int((lossdown&ok0[ii]&~good).sum()),'F_down_Z_down':int(bothdown.sum()),'F_down_Z_down_correct_to_wrong':int((bothdown&ok0[ii]&~good).sum()),'F_mean':float(F[ii,oi,si].mean()),'Z_mean':float(Z[ii,oi,si].mean()),'objective_mean':float(objectives[ii,oi,si].mean()),'mixed_objective_mean':float(mixed[ii,oi,si].mean()),'RMS_to_target_mean':float(np.sqrt(Z[ii,oi,si]).mean()),'RMS_to_target_median':float(np.median(np.sqrt(Z[ii,oi,si]))),'RMS_from_start_mean':float(np.sqrt(h[ii,oi,si,14]).mean()),'query_normalized_RMS_to_target_mean':float(np.sqrt(h[ii,oi,si,15]).mean())}
       for gi,g in enumerate(['Holding','In','drawer']):r.update({g+'_correct':int(gok[ii,oi,si,gi].sum()),g+'_margin_median':float(np.median(margin[ii,oi,si,gi])),g+'_margin_change_mean':float((margin[ii,oi,si,gi]-margin0[ii,gi]).mean())})
       summaries.append(r)
 save(E/'results/latent_summary.json',summaries);csvout(R/'latent_by_state_and_error.csv',summaries)
 endpoint=[]
 for i,c in enumerate(cases):
  for oi,obj in enumerate(['mixed','latent_only']):
   r=c|tags[i]|{'objective':obj,'baseline_correct':bool(ok0[i]),'final_correct':bool(ok[i,oi,-1]),'F_initial':float(f0[i]),'F_final':float(F[i,oi,-1]),'Z_initial':float(z0[i]),'Z_final':float(Z[i,oi,-1]),'objective_initial':float(objectives[i,oi,0]),'objective_final':float(objectives[i,oi,-1]),'RMS_target_final':float(np.sqrt(Z[i,oi,-1])),'RMS_move':float(np.sqrt(h[i,oi,-1,14]))}
   for gi,g in enumerate(['Holding','In','drawer']):r.update({g+'_initial_correct':bool(gok0[i,gi]),g+'_final_correct':bool(gok[i,oi,-1,gi]),g+'_initial_margin':float(margin0[i,gi]),g+'_final_margin':float(margin[i,oi,-1,gi])})
   endpoint.append(r)
 csvout(R/'latent_endpoints.csv',endpoint)
 # Margins are measured after computing unlabeled query JVP; no labels entered autograd.
 bm=np.where(y[:,:6],h[:,0,0,5:11],-h[:,0,0,5:11]);bd=np.where(y[:,:6],d['query_logit_direction_minus_gF'][:,:6],-d['query_logit_direction_minus_gF'][:,:6]);raw=h[:,0,0,11:14];cls=y[:,6:].argmax(1);comp=raw.copy();comp[np.arange(len(cases)),cls]=-np.inf;other=comp.argmax(1);qd=d['query_logit_direction_minus_gF'][:,6:]
 direc=np.stack([bd[np.arange(len(cases)),bm[:,:3].argmin(1)],bd[np.arange(len(cases)),3+bm[:,3:].argmin(1)],qd[np.arange(len(cases)),cls]-qd[np.arange(len(cases)),other]],1)
 gf=d['gF'].reshape(len(cases),-1);gz=d['g025Z'].reshape(len(cases),-1);nf=np.linalg.norm(gf,axis=1);nz=np.linalg.norm(gz,axis=1);gradient=[]
 for i,c in enumerate(cases):
  r=c|tags[i]|{'initial_correct':bool(ok0[i]),'F_initial':float(f0[i]),'weightedZ_initial':float(.25*z0[i]),'gradF_RMS':float(nf[i]/16),'grad025Z_RMS':float(nz[i]/16),'grad_norm_ratio_F_to_025Z':float(nf[i]/nz[i])if nz[i]>1e-12 else None,'grad_cosine_F_Z':float(np.dot(gf[i],gz[i])/(nf[i]*nz[i]))if nf[i]*nz[i]>1e-12 else None}
  for j,g in enumerate(['Holding','In','drawer']):r.update({g+'_initial_margin':float(margin0[i,j]),g+'_margin_direction_minus_gF':float(direc[i,j]),g+'_correct_margin_decreases_first_order':bool(gok0[i,j]and direc[i,j]<-eps)})
  for j,g in enumerate(['object_visual','object_geometry','scene_visual','robot']):r['weighted_'+g+'_gradient_RMS']=float(np.sqrt(np.square(d['gF_groups'][i,j]).mean()))
  gradient.append(r)
 csvout(R/'gradient_directions.csv',gradient)
 save(E/'results/gradient_records.json',gradient)
 # All adjacent frozen checkpoints, identical samples and query heads.
 cpy=ys[[r['sample_id']for r in cpm]];cok,cg,cm=semantics(cp[:,5:14],cpy);cf=cp[:,:4].mean(1);cz=cp[:,4];cl=cf+.25*cz;lookup={(r['model'],r['origin'],r['sample_id'],r['update']):i for i,r in enumerate(cpm)};pairs=[]
 for j,c in enumerate(cpm):
  if c['update']==0:continue
  i=lookup[(c['model'],c['origin'],c['sample_id'],c['update']-250)];s=samples[c['sample_id']]
  r={k:c[k]for k in ['model','seed','origin','sample_id']}|{k:s[k]for k in ['split','group','root_id','color','branch','global_step','skill']}|{'from_update':c['update']-250,'to_update':c['update'],'F_before':float(cf[i]),'F_after':float(cf[j]),'Z_before':float(cz[i]),'Z_after':float(cz[j]),'mixed_before':float(cl[i]),'mixed_after':float(cl[j]),'F_down':bool(cf[j]<cf[i]-eps),'Z_down':bool(cz[j]<cz[i]-eps),'mixed_down':bool(cl[j]<cl[i]-eps),'correct_before':bool(cok[i]),'correct_after':bool(cok[j]),'correct_to_wrong':bool(cok[i]and not cok[j]),'wrong_to_correct':bool(not cok[i]and cok[j])}
  for gi,g in enumerate(['Holding','In','drawer']):r.update({g+'_correct_before':bool(cg[i,gi]),g+'_correct_after':bool(cg[j,gi]),g+'_margin_before':float(cm[i,gi]),g+'_margin_after':float(cm[j,gi])})
  pairs.append(r)
 assert len(pairs)==7680;csvout(R/'checkpoint_pairs.csv',pairs);save(E/'results/checkpoint_pairs.json',pairs)
 ps=[]
 for seed in [0,1]:
  for origin in ['one_step','recursive']:
   for split in ['all','AB_train','C_development']:
    for branch in ['all','normal','held_closed']:
     rr=[r for r in pairs if r['seed']==seed and r['origin']==origin and(split=='all'or r['split']==split)and(branch=='all'or r['branch']==branch)]
     for cohort,pred in [('all',lambda r:True),('mixed_down',lambda r:r['mixed_down']),('F_down',lambda r:r['F_down']),('Z_down',lambda r:r['Z_down']),('F_and_Z_down',lambda r:r['F_down']and r['Z_down'])]:
      subset=[r for r in rr if pred(r)]
      if not subset:continue
      x={'seed':seed,'origin':origin,'split':split,'branch':branch,'cohort':cohort,'N_pairs':len(subset),'N_unique_roots':len({r['root_id']for r in subset}),'correct_before':sum(r['correct_before']for r in subset),'correct_after':sum(r['correct_after']for r in subset),'correct_to_wrong':sum(r['correct_to_wrong']for r in subset),'wrong_to_correct':sum(r['wrong_to_correct']for r in subset)}
      for g in ['Holding','In','drawer']:
       x[g+'_margin_delta_mean']=float(np.mean([r[g+'_margin_after']-r[g+'_margin_before']for r in subset]))
      ps.append(x)
 save(E/'results/checkpoint_summary.json',ps);csvout(R/'checkpoint_summary.csv',ps)
 save(E/'results/scoring_audit.json',{'scored_utc':now(),'all240_boundaries_valid_full9':True,'label_used_for_optimization':False,'label_used_for_sampling':False,'latent_endpoint_rows':len(endpoint),'gradient_rows':len(gradient),'latent_summary_groups':len(summaries),'checkpoint_adjacent_pairs':len(pairs),'checkpoint_summary_groups':len(ps),'margins':'binary signed raw logit;drawer true raw logit minus strongest rival;not calibrated confidence','actual_primary_N_per_seed':240,'no_independent_sample_claim':'Two queryseeds reuse observations; checkpoints/skill boundaries within roots dependent.'})
 print('SCORED',len(endpoint),len(pairs),len(summaries),flush=True)
if __name__=='__main__':main()
