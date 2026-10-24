from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.dpi':160})
def main():
 guard('report');rows=[];h0=[];fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True);colors={'original':'#777777','nominal':'#2878b5','recovery':'#c8552d'};names={'original':'原模型','nominal':'常规经验','recovery':'失败恢复经验'}
 for seed,ax in enumerate(axes):
  ax.fill_between([.5,1],0,.5,color='#e3f2e5',alpha=.7,label='理想判别区');ax.plot([0,1],[0,1],'--',c='gray',lw=1,label='两分支预测相同');ax.axhline(.5,c='black',lw=.7);ax.axvline(.5,c='black',lw=.7)
  for arm in ['original','nominal','recovery']:
   with np.load(S/f'results/predictions_{arm}_seed{seed}_evaluation.npz')as z:d={k:z[k]for k in z.files}
   with np.load(S/f'results/latent_{arm}_seed{seed}_evaluation.npz')as z:l={k:z[k]for k in z.files}
   pp=rel(d['prob']);yy=truth(d['cat']);ok=correct(pp,yy);ids=(d['mode']=='current')&(d['step']==0);h0.append({'seed':seed,'arm':arm,'held_correct':int(ok[ids,0].sum()),'InDrawer_correct':int(ok[ids,1].sum()),'drawer_correct':int(ok[ids,2].sum()),'boundaries':int(ids.sum())});xs=[];ys=[]
   for rid in sorted(set(d['root_id'])):
    ix=[np.where((d['root_id']==rid)&(d['branch']==b)&(d['step']==2)&(d['mode']=='current'))[0][0]for b in [1,2]];li=[np.where((l['episode_id']==str(rid)+f'_b{b}')&(l['step']==2))[0][0]for b in [1,2]];target=COLORS.index(str(d['target_color'][ix[0]]));x=float(pp[ix[0],target]);y=float(pp[ix[1],target]);eh=float(np.sqrt(np.mean((l['E'][li[0]].astype(float)-l['E'][li[1]].astype(float))**2)));hd=float(np.max(np.abs(l['h'][li[0]]-l['h'][li[1]])));qtv=float(np.abs(l['q'][li[0]].astype(float)-l['q'][li[1]].astype(float)).sum(-1).mean()/2);assert hd==0;xs.append(x);ys.append(y);rows.append({'seed':seed,'arm':arm,'root_id':str(rid),'target_color':str(d['target_color'][ix[0]]),'nominal_target_held_probability':x,'failed_target_held_probability':y,'target_probability_difference_failure_minus_success':y-x,'nominal_all_held_correct':bool(ok[ix[0],0]),'failure_all_held_correct':bool(ok[ix[1],0]),'paired_all_held_correct':bool(ok[ix,0].all()),'h2_E_RMS_between_actual_branches':eh,'h2_h_max_difference':hd,'h2_q_TV_between_actual_branches':qtv})
    if arm=='recovery':
     rank=sorted(set(d['root_id'])).index(rid);label=str(rid)[0]+str(rid)[-1]
     if x>.9:pos=(.68,.97-rank*.055)
     elif x<.1:pos=(.07,.04+rank*.05)
     else:pos=(x+.04,y+.035)
     ax.annotate(label,(x,y),xytext=pos,textcoords='data',fontsize=8,color=colors[arm],arrowprops={'arrowstyle':'-','color':colors[arm],'lw':.5})
   ax.scatter(xs,ys,c=colors[arm],s=55,alpha=.8,label=names[arm],marker={'original':'x','nominal':'s','recovery':'o'}[arm])
  ax.set_xlim(-.04,1.08);ax.set_ylim(-.04,1.08);ax.set_aspect('equal');ax.set_xlabel('同根真实成功分支：目标Held概率');ax.set_ylabel('同根真实落空分支：目标Held概率');ax.set_title(f'种子{seed}，每种模型6个独立配对根');ax.legend(fontsize=8,loc='center left');ax.grid(alpha=.2)
 fig.tight_layout();path=S/'figures/08_paired_observation.png';fig.savefig(path,bbox_inches='tight');plt.close(fig);csvwrite(S/'results/paired_observation_sensitivity.csv',rows);csvwrite(S/'results/initial_relation_errors.csv',h0)
 post=list(csv.DictReader((S/'results/posterior_updates.csv').open()));sums=[]
 for seed in [0,1]:
  rr=[r for r in post if r['seed']==str(seed)and r['variant']=='recovery'and r['split']=='evaluation'and r['branch']=='2'];sums.append({'seed':seed,'correct_failed_posterior':sum(r['posterior_held_correct']=='True'for r in rr),'already_correct_prior':sum(r['posterior_held_correct']=='True'and r['prior_held_correct']=='True'for r in rr),'wrong_to_correct_on_observation':sum(r['corrected_held']=='True'for r in rr),'paired_success_failure_both_correct':sum(r['paired_all_held_correct']for r in rows if r['seed']==seed and r['arm']=='recovery')})
 caption='横轴是同根真实抓取成功分支的目标Held概率，纵轴是实际落空分支的对应概率。灰叉、蓝方块、橙圆分别是原模型、常规经验组和恢复经验组；每种模型每面板有6个根，橙色A0–B2标出根。虚线表示两分支预测相同，黑色0.5线是固定阈值；右下浅绿区才是“成功判持有、失败判未持有”。大量点贴近对角线，说明准确率局部提高并不等于成功/失败条件区分能力。相同的h2确定性状态h与实际不同的观测编码E已核对；这张图不单独定位问题属于编码器、后验网络还是共享读出。'
 finding='更严格的配对证据不支持“已学会看到失败后纠错”：恢复组两个种子都没有任何根能同时辨对常规成功与真实落空（均0/6）。种子0落空Held判对的1根在收到h2观测前已经判对；种子1判对的3根中2根此前已判对，只有1根出现错→对，而该根的成功分支也被误判为空手。许多错误从h0的初始持有身份开始，不能把后续全部错误归因于抓取落空。'
 recommendation='因此本轮应记录为局部状态分数变化、未形成可靠的成败辨别和恢复预测，不据此进入可靠TS后继验收。已有真实对照数据可直接复用；如继续，先围绕h0持有身份泛化与相同历史下的观测条件区分确定一个可检验机制，再决定单项修正。此次不自行增加训练或改架构。'
 md='## 配对观测证据与首错位置\n\n'+finding+'\n\n![配对观测判别](figures/08_paired_observation.png)\n\n'+caption+'\n\n|种子|模型|h0持有身份正确/18|h0入屉集合正确/18|h0抽屉正确/18|\n|---|---|---|---|---|\n'+'\n'.join(f"|{r['seed']}|{r['arm']}|{r['held_correct']}|{r['InDrawer_correct']}|{r['drawer_correct']}|"for r in h0)+'\n\n'+recommendation+'\n'
 (S/'INTERPRETATION.md').write_text(md);p=S/'REPORT.md';text=p.read_text();head,body=text.split('\n',1);p.write_text(head+'\n\n'+finding+'\n\n'+body+'\n\n'+md)
 p=S/'REPORT.html';doc=p.read_text().replace('</h1>','</h1><p><strong>'+html.escape(finding)+'</strong></p>',1);ht='<table><tr><th>种子</th><th>模型</th><th>h0持有身份正确/18</th><th>h0入屉集合正确/18</th><th>h0抽屉正确/18</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in [r['seed'],r['arm'],r['held_correct'],r['InDrawer_correct'],r['drawer_correct']])+'</tr>'for r in h0)+'</table>';block='<h2>配对观测证据与首错位置</h2><p>'+html.escape(finding)+'</p><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><p>'+html.escape(caption)+'</p>'+ht+'<p>'+html.escape(recommendation)+'</p>';p.write_text(doc.replace('</html>',block+'</html>'));r=read(S/'results/report_complete.json');r['figures']=8;r['paired_evidence_added']=True;save(S/'results/report_complete.json',r);save(S/'results/interpretation.json',{'utc':now(),'finding':finding,'recommendation':recommendation,'recovery_prior_posterior_counts':sums,'h0':h0,'paired_rows':rows,'no_new_inference_or_training':True});print(finding,flush=True)
if __name__=='__main__':main()
