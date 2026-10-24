from common import *
def metrics(p,y,v,roots):
 pred=p>=.5;yy=y.astype(bool);vv=v.astype(bool);known=vv.all(1);result={'states':len(y),'roots':len(set(roots)),'all3_valid':int(known.sum()),'all3_correct':int(((pred==yy).all(1)&known).sum()),'colors':[]}
 result['all3_accuracy']=result['all3_correct']/result['all3_valid']if result['all3_valid']else None
 for j,c in enumerate(COLORS):
  pos=vv[:,j]&yy[:,j];neg=vv[:,j]&~yy[:,j];tp=int((pos&pred[:,j]).sum());fn=int((pos&~pred[:,j]).sum());fp=int((neg&pred[:,j]).sum());tn=int((neg&~pred[:,j]).sum())
  result['colors'].append({'color':c,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'FNR':fn/(tp+fn)if tp+fn else None,'FPR':fp/(tn+fp)if tn+fp else None,'positive_roots':len(set(roots[pos])),'negative_roots':len(set(roots[neg]))})
 be=[(c['FNR']+c['FPR'])/2 for c in result['colors']if c['FNR']is not None and c['FPR']is not None];result['mean_balanced_error']=float(np.mean(be))if be else None
 result['per_root']={str(r):{'valid':int((known&(roots==r)).sum()),'correct':int(((pred==yy).all(1)&known&(roots==r)).sum())}for r in sorted(set(roots))};return result

