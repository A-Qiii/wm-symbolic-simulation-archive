from common import *
def main():
 guard();summ=read(G/'results/probe_summary.json');rows=[r for r in summ if r['arm']=='geometry'];first=read(G/'results/first_errors.json');fr=[]
 for seed in [0,1]:
  for branch in [1,2]:
   rr=[r for r in first if r['arm']=='geometry'and r['seed']==seed and r['branch']==branch and r['template']=='recovery_seen6'and r['mode']=='free_h0'];counter=collections.Counter()
   for r in rr:
    p=np.array(r['predicted']);y=np.array(r['truth']);bad=[]
    if not np.array_equal(p[:3]>=.5,y[:3]):bad.append('Held')
    if not np.array_equal(p[3:6]>=.5,y[3:6]):bad.append('In')
    if p[6:].argmax()!=y[6:].argmax():bad.append('drawer')
    counter[(r['step'],r['skill'],','.join(bad))]+=1
   fr.extend({'seed':seed,'branch':branch,'step':k[0],'skill':k[1],'wrong_relations':k[2],'episodes':v}for k,v in counter.items())
 save(S/'results/previous_error_decomposition.json',{'probe_by_template':rows,'free_h0_first_errors':fr,'interpretation':'All48/46 failed non-target In intervals belong to cold-start four-step templates; familiar6 is48/48 for both seeds, with no previously-correct-to-wrong In event. This is largely the same initial relation failure, not evidence of an independent forgetting regression. Nominal six-step errors are still present: mainly repeated table-lift after the object is already InDrawer; perturbed branch errors at first injected grasp miss are not forecastable from identical public inputs.'});print(json.dumps(fr,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
