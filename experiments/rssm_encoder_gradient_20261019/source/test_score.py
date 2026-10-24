from score import *
def main():
 n=4;p=np.zeros((n,32));cat=np.zeros((n,5),dtype=int)
 for s in SLICES:p[:,s]=.1/(s.stop-s.start-1);p[:,s.start]=.9
 p[:,30:]=.1;p[0,0]=.01;p[0,1]=1-.01-6*.1/7
 base=cat.copy();base[0,0]=1
 x={'root_id':np.array(['r0','r1','r1','r1']),'prob':p,'cat':cat,'cat_valid':np.ones((n,5),bool),'binary':np.zeros((n,2),int),'binary_valid':np.ones((n,2),bool),'base_cat':base,'base_valid':np.ones((n,5),bool),'base_binary':np.zeros((n,2),int),'base_binary_valid':np.ones((n,2),bool),'event_prob':np.full((n,4),np.nan),'event_target':np.zeros((n,4)),'event_valid':np.zeros((n,4),bool),'role':np.array(['train']*n),'mode':np.array(['one_step']*n),'controls':np.array([0,1,1,1]),'status':np.array(['zero_control','success','success','success']),'skill':np.array(['open_drawer']*n),'all_prefix_actuated':np.zeros(n,bool)}
 r,_=score(x,{'split':'train','arm':'joint','seed':0})
 get=lambda f,c,m:next(z for z in r if z['stratum']=='all'and z['fact']==f and z['change']==c and z['metric']==m)
 a=get('red_location','all','accuracy');assert a['value']==.5 and a['micro_value']==.75
 assert get('red_location','changed','accuracy')['n']==1
 assert get('blue_location','changed','accuracy')['value']is None
 assert get('Holding_red','all','recall')['value']is None
 assert get('Holding_red','all','balanced_accuracy')['value']is None
 assert mean_ci([1,1])==(1.,1.,1.)
 out={'utc':now(),'passed':True,'tests':['Unequal root sizes: macro0.5 versus micro0.75','Changed fact denominator1, unchanged fact change denominator0','No positives: recall and BA undefined, not zero','Constant root bootstrap interval exact'],'optimizer_updates':0,'real_model_predictions_read':0,'note':'Initial ad-hoc fixture failed normalization assertion because its synthetic probability row summed below1; corrected fixture, no metric implementation change.'}
 save(E/'results/score_tests.json',out);print(out)
if __name__=='__main__':main()
