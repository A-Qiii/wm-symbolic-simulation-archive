"""Presentation revision: expose failed h0 training gate and avoid clipped labels."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 summary=read(S/'results/analysis.json')['summaries'];colors={'global':'#2675b7','object':'#e37b25'};plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(2,3,figsize=(15,8.7));ax=axs.flat[0]
 for seed in [0,1]:
  for arm in ARMS:
   rows=list(csv.DictReader((S/f'logs/{arm}_seed{seed}_train.csv').open()));loss=np.array([float(x['total'])for x in rows]);ax.plot(np.arange(50,len(loss)+1),np.convolve(loss,np.ones(50)/50,'valid'),color=colors[arm],linestyle='-'if seed==0 else'--',label=arm+' seed'+str(seed))
 ax.set(xlabel='Optimizer updates',ylabel='Common loss (50-update mean)',title='1. Training objective');ax.legend(fontsize=8);ax.grid(alpha=.2)
 panels=[('train','h0_full','2. Training h0 complete state: failed gate',False),('historical_evaluation','h0_full','3. Historical h0 complete state',False),('historical_evaluation','normal_whole','4. Historical normal whole trajectory',False),('historical_evaluation','primary_joint3','5. Historical paired current + 3 future steps',False),('historical_evaluation','forget_fixed_denominator_rate','6. Historical non-target InDrawer forgetting',True)]
 for ax,(split,k,title,forget)in zip(axs.flat[1:],panels):
  for j,arm in enumerate(ARMS):
   rr=[next(r for r in summary if r['split']==split and r['kind']=='role'and r['seed']==seed and r['arm']==arm)for seed in [0,1]];vals=[100*r[k]for r in rr];bars=ax.bar(np.arange(2)+(j-.5)*.32,vals,width=.30,color=colors[arm],label=arm)
   for bar,r,v in zip(bars,rr,vals):ax.text(bar.get_x()+bar.get_width()/2,v+(.15 if forget else 1.1),f"{r['forget_count']}/{r['stable_In_opportunities']}"if forget else f'{v:.1f}%',ha='center',fontsize=8)
  ax.set_xticks([0,1],['Seed 0','Seed 1']);ax.set(title=title,ylabel='Errors / fixed eligible opportunities (%)'if forget else'Family-macro accuracy (%)',ylim=(0,8 if forget else 108));ax.grid(axis='y',alpha=.2)
  if split=='train':ax.axhline(90,color='#555',ls=':',label='Required 90%')
  ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(S/'reports/comparison.png',dpi=160);fig.savefig(S/'reports/comparison.svg');plt.close(fig)
 report=S/'reports/REPORT.html';txt=report.read_text().replace('训练曲线实线为种子0、虚线为种子1。','训练曲线实线为种子0、虚线为种子1；第2图灰色点线是预定90%训练门槛。');report.write_text(txt)
if __name__=='__main__':main()
