from common import *
import torch,fcntl
from models import ObjectAug
from features import load
from evaluate import action,probabilities
from report import table
guard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
eps=read(S/'cache/episodes.json');by={e['episode_id']:e for e in eps};items=read(S/'cache/items.json');pairs=[];raw=np.load(S/'cache/geometry_raw.npz')
for e in eps:
 if e['template']!='order_known4':continue
 r=by[e['episode_id'].replace('order_known4','recovery_seen6')];i=e['boundaries'][0];j=r['boundaries'][1];exact=True
 for a,b in zip(items[i]['public_inputs'],items[j]['public_inputs']):
  with np.load(a)as x,np.load(b)as y:exact &= set(x.files)==set(y.files)and all(np.array_equal(x[k],y[k])for k in x.files)
 assert exact and all(np.array_equal(raw[k][i],raw[k][j])for k in ['objects','context'])
 x=np.load(items[i]['path']);y=np.load(items[j]['path']);pairs.append({'cold':e,'history':r,'cold_index':i,'history_index':j,'public_windows_exact':True,'raw_RGBD_geometry_exact':True,'same_selected_mask_ID_set':set(x['visible_registry_ids'])==set(y['visible_registry_ids']),'cold_mask_IDs':x['visible_registry_ids'].tolist(),'history_mask_IDs':y['visible_registry_ids'].tolist()})
primary_hashes={str(p):sha(p)for p in(S/'results').glob('predictions_*.npz')};rows=[]
for seed in [0,1]:
 for arm in ARMS:
  path=OLD/f'models/{arm}_seed{seed}_step1500.pt';m=ObjectAug().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);before=statehash(m);data=load(seed,arm)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for pair in pairs:
    e=pair['cold'];r=pair['history'];i=pair['cold_index'];j=pair['history_index'];k=r['boundaries'][0];h=COLORS.index(e['held_color']);assert read(e['rows'][0]['labels_path'])['before']['categorical'][h]==3
    cold=m.initial(data['objects'][i:i+1],data['context'][i:i+1],data['availability'][i:i+1]);history=m.initial(data['objects'][k:k+1],data['context'][k:k+1],data['availability'][k:k+1]);prior,_=m.imagine(history,*action('place_in_drawer'))
    # The sole intervention: keep the cold observation features exactly fixed;
    # replace the zero initial prior by the learned prior from a real public prefix.
    swapped=m.observe(prior,data['objects'][i:i+1],data['context'][i:i+1],data['availability'][i:i+1]);pc=probabilities(m,cold)[h*8+3];ps=probabilities(m,swapped)[h*8+3]
    rows.append({'arm':arm,'seed':seed,'family_id':e['family_id'],'episode_id':e['episode_id'],'branch':e['branch'],'H':e['held_color'],'raw_public_windows_exact':True,'selected_mask_sets_same':pair['same_selected_mask_ID_set'],'cold_H_In_probability':float(pc),'prior_swap_H_In_probability':float(ps),'true_H_In':True,'cold_correct':bool(pc>=.5),'prior_swap_correct':bool(ps>=.5),'object_feature_difference_cold_vs_history':float((data['objects'][i]-data['objects'][j]).abs().max()),'context_feature_difference_cold_vs_history':float((data['context'][i]-data['context'][j]).abs().max())})
  assert statehash(m)==before;del m,data;torch.cuda.empty_cache()
assert all(sha(p)==v for p,v in primary_hashes.items());verify()
csvwrite(S/'results/initial_relation_diagnostic.csv',rows);save(S/'results/initial_relation_diagnostic.json',{'utc':now(),'posthoc_diagnostic_not_primary_test':True,'intervention':'Only replace zero initial prior with frozen learned prior obtained from real recovery h0 plus requested place; preserve cold input features. No GT state/label injected.','public_window_pairs':len(pairs),'same_selected_mask_sets':sum(p['same_selected_mask_ID_set']for p in pairs),'front_context_confound':'Same raw images/depth/robot and raw explicit RGBD, but history-dependent12-slot registry can select different background masks; baseline comparison is therefore not a pure hidden-state contrast. The explicit prior swap holds cold encoded inputs fixed.','weights_and_original_test_predictions_unchanged':True,'original_cold_start_scores_not_replaced':True,'rows':rows})
summary=[]
for seed in [0,1]:
 for arm in ARMS:
  rr=[r for r in rows if r['seed']==seed and r['arm']==arm];summary.append([seed,arm,len(rr),sum(r['cold_correct']for r in rr),sum(r['prior_swap_correct']for r in rr)])
block='<h2>首错定位：已有In关系的冷启动，而非先归咎于新顺序</h2><p>两个四步模板的全部24条轨迹、四个模型都在h0首次发生关系错误；此时尚未执行新顺序。与六步恢复链的h1配对，24组四帧RGB-D/机器人观测逐字节相同，原显式RGB-D几何也完全相同。但是12槽可见区域注册器带有历史，不同初始化可能选择不同背景掩码，冻结前端特征因此并不完全相同。不能仅凭相同图像就断言只有递归状态不同。</p>'+table(['种子','组','匹配初态','原冷启动H-In读对','仅替换历史先验后H-In读对'],summary)+'<p>上述事后诊断只替换一个因素：保留冷启动全部观测特征，用真实公开前缀（持有H的h0和一次place请求）产生的冻结学习先验，替换零初始先验。没有注入GT关系、没有训练、没有改阈值，也不把诊断成绩替换原测试的0分。它检验真实历史提供的先验能否补足初态In读出；即使恢复，也不能证明模型已经从单独观测学会任意已有In关系，更不能据此宣称新顺序通过。两个四步整轨的地板结果限制了本轮对顺序本身的归因。</p>'
p=S/'reports/REPORT.html';text=p.read_text().replace('<h2>相同初态与动作集合的新顺序</h2>',block+'<h2>相同初态与动作集合的新顺序</h2>');p.write_text(text)
print(json.dumps(summary),flush=True)
