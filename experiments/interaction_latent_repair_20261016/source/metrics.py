from common import *
def discrete(p):
 a=p>=.5;a[:,6:]=False;a[np.arange(len(p)),6+p[:,6:].argmax(1)]=True;return a
def score(pred,y,v,roots):
 pred=discrete(pred);y=np.array(y,bool);v=np.array(v,bool);ok=((pred==y)|~v);valid=v.any(1);full=ok.all(1)&valid;out={'states':len(y),'roots':len(set(roots)),'valid_facts':int(v.sum()),'correct_facts':int(((pred==y)&v).sum()),'all_valid_facts_correct_states':int(full.sum()),'root_macro_all_facts_accuracy':float(np.mean([full[np.array(roots)==r].mean()for r in set(roots)]))if roots else None,'all_states_correct_roots':sum(bool(full[np.array(roots)==r].all())for r in set(roots))}
 for lo,hi,name in[(0,3,'Holding'),(3,6,'In'),(6,9,'drawer')]:
  vv=v[:,lo:hi];yy=y[:,lo:hi];pp=pred[:,lo:hi];out[name]={'correct':int(((pp==yy)&vv).sum()),'valid':int(vv.sum()),'TP':int((pp&yy&vv).sum()),'FP':int((pp&~yy&vv).sum()),'FN':int((~pp&yy&vv).sum())}
 return out
