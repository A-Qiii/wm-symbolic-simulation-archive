"""Descriptive post-evaluation diagnosis only; no model change or metric selection."""
from common import *
import html
def main():
 guard();rows=[];summary=[];names=['table_left','table_middle','table_right','drawer','slider_left','slider_right','held','other']
 for arm in ARMS:
  for seed in [0,1]:
   z=dict(np.load(S/f'results/predictions_{arm}_seed{seed}_train.npz'));pred=z['prob'][:,:24].reshape(-1,3,8).argmax(-1);mask=(z['step']==0)&(z['mode']=='current')
   for i in np.where(mask)[0]:
    for j in range(3):
     t=int(z['cat'][i,j]);p=int(pred[i,j])
     if z['cat_valid'][i,j]and t!=p:rows.append({'arm':arm,'seed':seed,'kind':str(z['kind'][i]),'root_id':str(z['root_id'][i]),'family_id':str(z['family_id'][i]),'branch':int(z['branch'][i]),'object':COLORS[j],'truth':names[t],'prediction':names[p],'table_band_only':bool(t<3 and p<3),'truth_probability':float(z['prob'][i,8*j+t]),'predicted_probability':float(z['prob'][i,8*j+p]),'visible_fraction':float(z['availability'][i,j])})
   for kind in ['core','role']:
    rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['kind']==kind];summary.append({'arm':arm,'seed':seed,'kind':kind,'wrong_object_boundary_records':len(rr),'table_band_only':sum(r['table_band_only']for r in rr),'other_category_errors':sum(not r['table_band_only']for r in rr),'distinct_roots_with_errors':len({r['root_id']for r in rr})})
 csvwrite(S/'results/h0_category_errors.csv',rows);save(S/'results/h0_diagnosis.json',{'utc':now(),'posthoc_descriptive_only':True,'no_new_inference_or_training':True,'counts_are_object_boundary_records_not_independent_samples':True,'summary':summary})
 paragraph='<p><strong>最早未满足门槛的错误已经在h0：</strong>对象模型训练角色的完整状态准确率为62.78%/83.06%，低于90%；种子1旧训练根为67.50%。主要错误是桌面左/中/右位置分类混淆。桌面三区在Held/InDrawer关系上均为“否”，所以这些错误可能被联合关系指标掩盖；完整h0门槛仍必须判失败。这解释了“历史联合改善”与“拟合门槛失败”能够同时出现，不能据此认定单一结构性原因。</p>'
 report=S/'reports/REPORT.html';text=report.read_text();text=text.replace('<h2>1. 先检查训练拟合</h2>','<h2>1. 先检查训练拟合</h2>'+paragraph);text+='<h2>补充：h0类别错误的实际构成</h2><p>以下计数为对象×分支边界记录，配对/重复分支不能当独立样本。与正式家族统计分开，仅描述已发生错误，不更改门槛。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>';report.write_text(text)
 with (S/'reports/RESULT.md').open('a')as f:f.write('\n最早拟合缺口在h0，主要为桌面左/中/右分类；这些类别在Held/InDrawer关系上相同，因此联合关系改善并不消除完整状态门槛失败。详细逐对象错误见results/h0_category_errors.csv。\n')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
