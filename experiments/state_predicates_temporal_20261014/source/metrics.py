import numpy as np
GROUPS={'held':[0,1,2],'in_drawer':[3,4,5],'open_drawer':[6]}
def rate(n,d):return float(n/d) if d else None
def known(p,taus):
 t=np.array([taus[0]]*3+[taus[1]]*3+[taus[2]]);k=(p>=t)|(p<=1-t);conflict=(p[:,:3]>=t[:3]).sum(1)>1;k[:,:3]&=~(conflict[:,None]&(p[:,:3]>=t[:3]));return k
def metric(p,y,v,k):
 pp=p[v];yy=y[v].astype(bool);kk=k[v];hat=pp>=.5;tp=int((hat&yy).sum());tn=int((~hat&~yy).sum());fp=int((hat&~yy).sum());fn=int((~hat&yy).sum());pos=int(yy.sum());neg=len(yy)-pos;err=hat!=yy;conf=np.maximum(pp,1-pp)
 return {'n':len(yy),'positive':pos,'negative':neg,'TP':tp,'TN':tn,'FP':fp,'FN':fn,'accuracy':rate(tp+tn,len(yy)),'balanced_accuracy':(tp/pos+tn/neg)/2 if pos and neg else None,'raw_positive_predictions':int(hat.sum()),'positive_recall':rate(tp,pos),'FPR':rate(fp,neg),'known':int(kk.sum()),'unknown':int((~kk).sum()),'coverage':rate(kk.sum(),len(yy)),'known_positive_predictions':int((hat&kk).sum()),'known_true_positive':int((hat&yy&kk).sum()),'known_false_positive':int((hat&~yy&kk).sum()),'known_false_negative':int((~hat&yy&kk).sum()),'known_error':rate((err&kk).sum(),kk.sum()),'positive_UNKNOWN':int((yy&~kk).sum()),'negative_UNKNOWN':int((~yy&~kk).sum()),'wrong_confidence':conf[err].tolist(),'wrong_probability_true':pp[err].tolist(),'Brier':float(np.mean((pp-yy)**2)) if len(yy) else None}
def summarize(p,y,v,taus):
 k=known(p,taus);return {name:metric(p[:,cols],y[:,cols],v[:,cols],k[:,cols]) for name,cols in GROUPS.items()}
def calibrate(p,y,v):
 taus=[.5,.5,.5];details={}
 for gi,(name,cols) in enumerate(GROUPS.items()):
  if name=='in_drawer':details[name]={'threshold':.5,'status':'retained_old_threshold_no_new_calibration'};continue
  grid=[];winner=None
  for t in [.5,.6,.7,.8,.85,.9,.95,.975,.99,.995]:
   ts=[.5,.5,.5];ts[gi]=t;k=known(p,ts);m=metric(p[:,cols],y[:,cols],v[:,cols],k[:,cols]);ke=m['known_error'];eligible=m['known']>=20 and ke is not None and ke<=.05 and m['known_true_positive']+m['known_false_negative']>=1 and m['known']-(m['known_true_positive']+m['known_false_negative'])>=1
   if name=='held':eligible=eligible and m['known_false_positive']/max(m['negative'],1)<=.01
   grid.append({'threshold':t,'eligible':eligible,**m})
   if eligible and (winner is None or m['known']>winner['known']):winner={'threshold':t,**m}
  taus[gi]=winner['threshold'] if winner else 1.000001;details[name]={'selected':winner,'all_UNKNOWN':winner is None,'grid':grid}
 return {'thresholds':taus,'details':details}
