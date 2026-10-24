from experiment import *
import os,csv,html,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
LABEL={'full_seed0':'转移 seed 0','full_seed1':'转移 seed 1','full_seed2':'转移 seed 2','copy_effect':'复制＋AB效果','persistence':'保持当前状态','real_next_observation':'真实后观测读出'}
PALETTE={'full_seed0':'#587cb1','full_seed1':'#9175ba','full_seed2':'#2d5d88','copy_effect':'#dc8a35','persistence':'#5c9188','real_next_observation':'#9ca8b1'}
FAM={'close':'关抽屉','lift':'提起指定物','place':'放入抽屉'}
COLOR={'red':'红','blue':'蓝','pink':'粉'}
COND={'empty_open':'空手、抽屉开','held_open':'手持目标物、抽屉开','empty_closed':'空手、抽屉关','empty_target_table':'空手、目标在桌面','held_other_target_table':'手持另一物、目标在桌面','held_same_target':'已持指定目标物','held_closed':'手持目标物、抽屉关'}
ROLE={'positive_reference':'可执行参考','core_state_dependence':'关键条件对照','nochange_semantics':'不变语义对照'}
MODELS=['full_seed0','full_seed1','full_seed2','copy_effect','persistence','real_next_observation']
FACT=['持红','持蓝','持粉','红入屉','蓝入屉','粉入屉','抽屉开']
def esc(x):return html.escape(str(x))
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table></div>'
def csvout(name,rows):
 with (E/'results'/name).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio(a,b):return f'{a}/{b}'
def short(r):return r['environment']+'·'+FAM[r['family']]+'·'+COLOR[r['target_color']]+'·'+str(r['replicate'])
def main():
 assert read(E/'results/final_validation.json')['passed'];verify_frozen();ev=read(E/'results/evaluation.json');roots=read(B/'roots_manifest.json');manifest=read(B/'data_manifest.json');pairs=read(E/'data/public/pairs_full.json');windows=read(E/'data/public/windows_full.json');wm={w['window_id']:w for w in windows};pid={r['episode_id']:i for i,r in enumerate(pairs)};priv={r['pair_id']:r for r in manifest['private_pairs']};contrasts=read(E/'results/root_contrasts.json');pred=dict(np.load(E/'results/predictions.npz'));t=dict(np.load(E/'data/_private/supervision_full.npz'));detail={(r['model'],r['episode_id']):r for r in ev['details']};figdir=E/'figures'
 # Main chart: denominator and independence explicitly shown, reference separated by hatch.
 fig,ax=plt.subplots(1,2,figsize=(14,5.8),gridspec_kw={'width_ratios':[1.2,1]});ys=np.arange(len(MODELS))
 vals=[ev['results'][n]['C']['all']['all7_known_correct_pairs']for n in MODELS]
 for i,n in enumerate(MODELS):
  ax[0].barh(i,vals[i],color=PALETTE[n],height=.64,hatch='///'if n=='real_next_observation'else None);ax[0].text(vals[i]+.25,i,f'{vals[i]}/18',va='center')
  v=sum(r['models'][n]['both_target_effect_facts_correct']for r in contrasts if r['split']=='dev');ax[1].barh(i,v,color=PALETTE[n],height=.64,hatch='///'if n=='real_next_observation'else None);ax[1].text(v+.10,i,f'{v}/6',va='center')
 for a in ax:a.set_yticks(ys,labels=[LABEL[n]for n in MODELS]);a.invert_yaxis();a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
 ax[0].set_xlim(0,20);ax[0].set_xticks([0,3,6,9,12,15,18]);ax[0].set_xlabel('七事实全部已知且正确的转移数（18次，来自6个C根）');ax[0].set_title('全部尝试都计入：UNKNOWN不算正确')
 ax[1].set_xlim(0,7);ax[1].set_xticks(range(7));ax[1].set_xlabel('同根两侧目标事实都正确的根数（6个C根）');ax[1].set_title('可执行 b0 ＋关键条件 b1：两侧必须同时正确')
 fig.suptitle('冻结模型的新条件诊断：会预测成功轨迹，不代表能分辨可执行条件',fontsize=15);fig.text(.02,.02,'颜色对应左侧方法名称；斜线灰条使用真实未来观测，只是读出参考，不是可部署预测器。两图计量单位不同，根内3次不视为3个独立根。',fontsize=9);fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(figdir/'C_comparison.png',dpi=160);plt.close(fig)
 # Every root, no success-only subset.
 mat=np.array([[sum(detail[(n,b['episode_id'])]['all7_correct']for b in r['branches'])for n in MODELS]for r in roots]);fig,ax=plt.subplots(figsize=(11.5,9));im=ax.imshow(mat,cmap=ListedColormap(['#fae8d9','#edc399','#90b7af','#23675d']),vmin=0,vmax=3,aspect='auto')
 ax.set_xticks(range(len(MODELS)),labels=[LABEL[n]for n in MODELS],rotation=15,ha='right');ax.set_yticks(range(len(roots)),labels=[short(r)for r in roots]);ax.axhline(11.5,color='#283b4c',lw=2)
 for i in range(len(roots)):
  for j in range(len(MODELS)):ax.text(j,i,str(mat[i,j])+'/3',ha='center',va='center',color='white'if mat[i,j]==3 else'#273949')
 cb=fig.colorbar(im,ax=ax,ticks=[0,1,2,3],shrink=.75);cb.set_label('每根3次中，七事实全部已知且正确的次数');ax.set_title('全部18个根｜上方A/B用于未来训练，下方C仅开发诊断');ax.set_xlabel('方法；最后一列读取真实后观测，不是预测成绩');ax.set_ylabel('独立根：环境·动作族·目标色·重复编号');fig.tight_layout();fig.savefig(figdir/'all_roots.png',dpi=160);plt.close(fig)
 # Cases chosen by plan order, not model outcome.
 selected=[next(r for r in roots if r['split']=='dev'and r['family']==f)for f in ['close','lift','place']];cases=[]
 for r in selected:
  fig=plt.figure(figsize=(14,10.5));outer=fig.add_gridspec(1,2,wspace=.32,left=.09,right=.97,bottom=.12,top=.87)
  for bi in [0,1]:
   ep=r['branches'][bi]['episode_id'];i=pid[ep];p=pairs[i];pr=priv[p['pair_id']];gs=outer[0,bi].subgridspec(3,2,height_ratios=[1,1,1.55],wspace=.08,hspace=.3)
   for row,end in enumerate(['before','after']):
    w=wm[p[end+'_window_id']];z=np.load(w['public_frames'][-1]);tm=w['simulation_times'][-1]
    for col,cam in enumerate(['rgb_static','rgb_gripper']):
     a=fig.add_subplot(gs[row,col]);a.imshow(z[cam],interpolation='nearest');a.axis('off');a.set_title(('前'if row==0 else'后')+('·外相机'if col==0 else'·腕相机')+f'（{tm:.2f}s）',fontsize=10)
   ys=[t['before_y'][i].astype(int).tolist(),detail[('persistence',ep)]['prediction_ternary'],t['after_y'][i].astype(int).tolist(),detail[('full_seed2',ep)]['prediction_ternary'],detail[('copy_effect',ep)]['prediction_ternary'],detail[('real_next_observation',ep)]['prediction_ternary']]
   cell=np.zeros((6,7),int)
   for row in [1,3,4,5]:
    truth=ys[0]if row==1 else ys[2]
    for j,val in enumerate(ys[row]):cell[row,j]=3 if val=='UNKNOWN'else 1 if val==truth[j]else 2
   a=fig.add_subplot(gs[2,:]);a.imshow(cell,cmap=ListedColormap(['#e0eaf2','#c2e1d8','#f2b78d','#c5c8cc']),vmin=0,vmax=3,aspect='auto');a.set_xticks(range(7),labels=FACT,fontsize=9);a.set_yticks(range(6),labels=['真实前事实','当前读出','真实后事实','转移 seed 2','复制＋效果','真实后读出'],fontsize=9);a.set_title('七项事实逐项核对',fontsize=10)
   for row in range(6):
    for j,val in enumerate(ys[row]):a.text(j,row,'?'if val=='UNKNOWN'else str(val),ha='center',va='center',fontsize=11)
   xpos=.30 if bi==0 else .77;fig.text(xpos,.91,f'b{bi} {ROLE[pr["contrast_role"]]}｜{COND[pr["condition"]]}\n实际：'+('成功、事实改变'if pr['outcome_group']=='success_changed'else'控制器拒绝、事实不变'),ha='center',fontsize=11)
  fig.suptitle(FAM[r['family']]+'｜同一个新根、同一个声明动作\n'+r['root_id'],fontsize=14,y=.985)
  fig.text(.04,.05,'表中1=真、0=假、?=UNKNOWN；蓝灰=真实标签，绿=与对应时刻标签一致，橙=错误，灰=UNKNOWN。\n图片均为本次实测锚点帧（放大显示）；模型实际输入每端4帧、跨度2.1秒。上方秒数为各分支自己的仿真时钟。\n右侧拒绝后仍执行96步真实保持；该不变结果描述固定控制器接口，不代表物体自然无法运动。',fontsize=10)
  name='case_'+r['family']+'.png';fig.savefig(figdir/name,dpi=160);plt.close(fig);cases.append({'root_id':r['root_id'],'family':r['family'],'file':name,'selection':'first C root in frozen plan per family, branches b0 and b1; no outcome filtering'})
 save(E/'results/figure_cases.json',cases)
 # Compact human and machine tables.
 prep=[];attempts=[]
 for ep in manifest['episodes']:
  for a in ep['preparation_attempts']:prep.append({'episode_id':ep['episode_id'],'split':ep['split'],'family':ep['family'],'condition':ep['condition'],'preparation_index':a['index'],'skill':a['skill'],'status':a['result']['status'],'reason':a['result']['reason'],'controls':a['result']['controls'],'attempts':a['result']['attempts'],'retries':a['result']['retries']})
  pp=next(p for p in pairs if p['episode_id']==ep['episode_id']);a=priv[pp['pair_id']];attempts.append({'root_id':ep['root_id'],'episode_id':ep['episode_id'],'split':ep['split'],'family':ep['family'],'condition':ep['condition'],'prep_count':len(ep['preparation_attempts']),'covered':ep['covered'],'uncovered_reason':ep['uncovered_reason'],'diagnostic_skill':pp['skill'],'status':a['primitive']['status'],'reason':a['primitive']['reason'],'primitive_controls':a['primitive']['controls'],'posthold_controls':a['posthold']['controls'],'outcome':a['outcome_group'],'truth_before':json.dumps(a['before_targets']['y']),'truth_after':json.dumps(a['after_targets']['y'])})
 csvout('preparation_attempts.csv',prep);csvout('all_collection_attempts.csv',attempts)
 rootrows=[]
 for r in roots:
  row={'root_id':r['root_id'],'split':r['split'],'dev_role':r['dev_role'],'family':r['family'],'colour':r['target_color'],'calls':3,'success_changed':1,'refused_nochange':2}
  for n in MODELS:row[n+'_all7']=sum(detail[(n,b['episode_id'])]['all7_correct']for b in r['branches']);row[n+'_UNKNOWN']=sum(len(detail[(n,b['episode_id'])]['UNKNOWN_columns'])for b in r['branches'])
  rootrows.append(row)
 csvout('root_summary.csv',rootrows)
 # Evidence for limited training-data coverage, from AB only.
 oldmeta=read(OLD/'data/public/pairs.json');ot=np.load(OLD/'data/_private/training_targets_and_audit.npz');coverage=[]
 for skill in sorted({r['skill']for r in oldmeta}):
  ids=[i for i,r in enumerate(oldmeta)if r['split']=='train'and r['skill']==skill];coverage.append({'skill':skill,'old_AB_pairs':len(ids),'before_patterns':np.unique(ot['before_y'][ids].astype(int),axis=0).tolist(),'after_patterns':np.unique(ot['after_y'][ids].astype(int),axis=0).tolist()})
 save(E/'results/old_AB_action_state_coverage.json',coverage)
 proposal={'status':'proposal_only_not_executed','purpose':'Test whether adding the finite state-condition examples repairs successor prediction with the same architecture; no independent-test or full-guide acceptance claim.', 'evidence':{'current_reader_C_before_after':'18/18 each','full_three_seeds_C_all7':[4,4,4],'copy_effect_C_all7':10,'full_and_copy_core_contrast_roots_C':'0/6','single_replacement':'actual next latent + same frozen head =>18/18'},'trainable_scope':'Only existing 76224-parameter full action-conditioned latent transition. Freeze DINO, RGB registry v1, normalization, current-state encoder/GRUs, symbol heads, thresholds, controller, labels, and copy-effect rules.','data':{'AB_old_pairs':96,'AB_new_pairs':36,'AB_total_pairs':132,'AB_independent_roots':36,'AB_new_nochange_pairs':24,'C_old_pairs':24,'C_new_pairs':18,'C_total_pairs':42,'C_independent_roots':12,'C_tune_pairs':21,'C_calibration_pairs':21,'retain_all_attempts':True,'no_C_in_gradient_or_normalization':True,'no_D':True},'bounded_suggestion':{'full_seeds':[0,1,2],'updates_each':600,'total_update_cap':1800,'initialization':'same original random-seed initialization, not a continuing optimization sweep; one protocol','batch_size':16,'sampling':'uniform permutations of all132 AB pairs, seed-fixed; no new resampling or success filtering','optimizer':'same AdamW lr0.001 weight_decay0.0001 grad_clip1','loss':'same three-group AB-weighted BCE +0.05 mean object/scene SmoothL1; class weights recomputed from132 AB labels with the same formula, never C','architecture_or_threshold_change':False,'budget_extension':False},'reporting':'Pre-register before any update: all3seeds, old/new C separately and root-macro overall; exact matched-root both-sides correctness, changed/unchanged/other entities, UNKNOWN, all7, before-reader subsets, latent MAE/range. Keep frozen copy-effect and persistence comparisons. Any checkpoint selection only predeclared C-tune, no new seed search.','limits':'C is already viewed development data. Preserve original acceptance records; do not redefine an impossible saturated changed-fact comparison as a new success. This proposed training is a bounded diagnostic, not authorization for D, longer rollout, TS/LTL closure, more collection or changes to the architecture. If it fails, report and stop.'}
 save(E/'NEXT_TRAINING_PROPOSAL.json',proposal)
 # Report content: all numerical claims drawn from saved results.
 body=[];add=body.append
 add('<div class="eyebrow">本机实验 · 冻结诊断 · '+now()[:10]+'</div><h1>加入状态条件后，原转移成绩未能保持</h1><p class="lead">18个新根、54次有限诊断全部完成。新C的前后观测读出各18/18正确，但三个冻结转移模型均只有4/18次七事实全对；复制＋AB效果为10/18。六组C关键对照中，两者都没有同时预测对可执行与拒绝两侧。</p>')
 add('<div class="note">本轮没有训练、调阈值、改规则、换控制器、采集D、运行闭环或上传。18个成功与36个拒绝结果全部保留。拒绝后的不变描述<strong>固定控制器的接口行为</strong>，不是自然物理不可执行性的证明。</div>')
 add('<h2>采了什么，为什么这样配对</h2><p>A/B合计12个新根、36次；C为6个新根、18次。每根从完全相同初始布局和种子复位三次，分别准备不同条件后调用同一声明动作。三分支仍只算一个独立根。每个划分在三类动作间均衡，AB每色4根、C每色2根；C每个动作族仅覆盖两色，尚不是三色×三动作的完整组合。</p>')
 rr=[]
 for fam in ['close','lift','place']:
  r=next(r for r in roots if r['family']==fam)
  for b in r['branches']:
   condition=COND[b['condition']]+('、抽屉关'if fam=='lift'else'');rr.append([FAM[fam],f'b{b["branch"]} '+ROLE[b['role']],condition,' → '.join(b['preparation_skills'])or'无准备技能',('成功、改变（6/6）'if b['branch']==0 else'拒绝、不变（6/6）')])
 add(table(['声明动作族','分支用途','调用前条件','合法准备（以一根颜色为例）','实际诊断结果'],rr));add('<p>准备调用总计54次，全部成功、零重试；诊断总计54次，18次成功、36次正常拒绝、零失败、零未覆盖。若准备失败，原计划要求保留并记未覆盖；本轮没有发生，也没有补采替换。每个准备／诊断原语预算≤900控制步，拒绝原语自身0步，但仍追加96步真实保持。</p>')
 add('<h2>所有比较都保留七项事实与UNKNOWN</h2><p>七事实为：持有红／蓝／粉物、红／蓝／粉物在抽屉中、抽屉打开。held依原双指接触与支撑规则，in依原抽屉归属规则，open依原标注阈值；全部378个转移事实在前后端有效。七事实全对要求七项均已知且正确；UNKNOWN不计正确。未变事实分母包含拒绝转移中的所有事实，也包含成功转移中未改变的事实。</p>')
 rows=[]
 for n in MODELS:
  a=ev['results'][n]['AB']['all'];c=ev['results'][n]['C']['all'];m=c['mask_metrics'];rows.append([LABEL[n],ratio(a['all7_known_correct_pairs'],36),ratio(c['all7_known_correct_pairs'],18),ratio(m['changed']['strict_known_correct'],m['changed']['facts']),ratio(m['unchanged']['strict_known_correct'],m['unchanged']['facts']),ratio(m['all7']['UNKNOWN'],126),ratio(c['all_pairs_correct_roots'],6)])
 add(table(['方法','新AB七事实全对','新C七事实全对','C改变事实正确','C未变事实正确','C UNKNOWN事实','C三个分支全对根'],rows));add('<p>真实后观测读出读取真实未来图像，只是测量参照。保持当前状态因本轮2/3尝试被拒绝而达到12/18，但全部6次成功改变均失败，不能当成学会动作效果。复制效果在成功改变事实上8/8正确，却不能识别关键适用条件。三种子并非三个独立数据集；C只有六个独立根，未用18次当成18个独立样本做置信推断。</p>')
 add('<figure><img src="figures/C_comparison.png"><figcaption>左图横轴是18次C转移中七事实全对的次数；右图横轴是6个C根中，可执行与关键条件两侧的目标效果事实都正确的根数。颜色标识方法，灰色斜线表示使用真实后观测的参照。两侧的声明动作及目标事实初值一致、实际后值不同，所以只套用无条件成功效果不能同时解释两侧。</figcaption></figure>')
 rows=[]
 for n in MODELS:
  r=ev['results'][n]['C'];rows.append([LABEL[n]]+[ratio(r[s]['all7_known_correct_pairs'],r[s]['pairs'])for s in ['role:positive_reference','role:core_state_dependence','role:nochange_semantics','family:close','family:lift','family:place']])
 add(table(['方法','可执行参考','关键条件对照','不变语义对照','关抽屉','提起','放入'],rows));add('<p>复制基线的16个C UNKNOWN来自两次“已持另一物再提起”产生持有冲突（4项），以及两次“空手放入”无法绑定目标（12项）。三个转移模型的UNKNOWN均为0，错误仍大量存在；“没有UNKNOWN”本身不表示可靠。</p>')
 add('<h2>第一个有影响的错误在哪里</h2><p>全部54次的当前与真实后观测读出均为七事实全对，没有当前读出错误子集。本轮14个C转移错误全部落在“双端读出正确”子集。按冻结清单，首个C错误是 <code>C_statedep_close_pink_0__b0</code>：抽屉实际关上，seed 2却以0.999107概率预测仍打开。</p><p>仅将预测后继隐状态替换为真实后观测的编码、保持同一个冻结符号头，C从4/18变为18/18。这是离线单项替换诊断，真实未来没有输入预测器；结果定位到后继预测路径，尚不能唯一拆分“转移映射”与“隐表示对预测是否充分”。没有据此修改模型。</p>')
 add('<p>动作编码已与旧训练逐项核对一致。旧AB的全部24次提起都发生在抽屉已开时，24次关抽屉都发生在有物体已入屉时；旧96次全为成功轨迹。新数据中，提起发生在抽屉关闭时，关抽屉可发生在空抽屉或持物时。seed 2把新C全部18个后继都预测为抽屉打开，且会在持物拒绝后虚构物体入屉。这与旧数据的状态—阶段相关性一致，是覆盖不足的证据；不是证明模型内部采用了某一条唯一规则。</p>')
 add('<h2>三类同根实测对照</h2><p>每类取冻结计划中第一个C根的b0/b1，选择不依据预测成败。图片只显示每端锚点帧；模型实际使用每端四帧。表中完整保留七项，避免只看动作目标而漏掉被错误改变的其他事实。</p>')
 for c in cases:add('<figure><img src="figures/'+c['file']+'"><figcaption>'+esc(c['root_id'])+'：左侧为可执行参考，右侧为关键条件对照。列为持有三色、三色入屉和抽屉开；1真、0假、?未知。绿色预测与该时刻真实标签一致，橙色错误，灰色未知，蓝灰行是真实标签。外相机200×200、腕相机84×84均来自原始实测，未生成未来图像。</figcaption></figure>')
 add('<h2>逐根结果与数据边界</h2><figure><img src="figures/all_roots.png"><figcaption>每行一个独立根，每格为三个分支中七事实全对的次数。橙浅色0到深绿3表示正确次数增加；横线分开A/B训练划分与C开发划分。最后一列是真实后观测参照，不能与预测器混为部署性能。</figcaption></figure>')
 rows=[]
 for r in rootrows:rows.append([r['root_id'],r['split'],r['dev_role']or'—']+[str(r[n+'_all7'])+'/3'for n in MODELS[:4]]+[str(r['copy_effect_UNKNOWN'])])
 add(table(['独立根','划分','C用途','转移0','转移1','转移2','复制效果','复制UNKNOWN项'],rows))
 add('<p>AB是将来可训练数据，本轮尚未训练；C已用于多轮开发观察，不是盲测。C tune与calibration各3根、9次，本轮没有用它们重新选模型、阈值或规则。18组同根关键对照（其中C六组）都满足同声明动作、目标事实相同初值而后值不同；但不同准备也改变机器人姿态等状态，不能把它们称为只改变单个物理因子的严格因果干预。</p>')
 rows=[]
 for n in MODELS[:3]:
  m=ev['latent_metrics'][n]['C'];rows.append([LABEL[n],f'{m["object_MAE"]:.4f}',f'{m["scene_MAE"]:.4f}',f'{100*m["outside_GRU_range"]:.2f}%'])
 add(table(['新C后继隐状态','物体平均绝对误差','场景平均绝对误差','坐标超出GRU[-1,1]比例'],rows));add('<p>误差相对同一冻结编码器的真实后继隐状态计算；单位是隐状态数值，不是米。超范围说明预测落到编码器输出范围外。单步七事实即使改善，也不能直接推断多步滚动或规划已可靠。</p>')
 add('<h2>建议下一次只训练什么</h2><p>当前证据支持一次有限的数据覆盖修正实验：只重训原76,224参数的完整转移模块。旧AB96＋新AB36＝132对、36个根；包含新24次拒绝不变，全部保留。旧C24＋新C18＝42对、12根继续仅开发评估，分别报告旧／新数据，按根汇总。</p><p>建议一次协议固定三个原种子、每种子600更新、总计1800更新，上限不扩展；沿用原初始化、结构、AdamW参数、批量16和损失形式，按132个AB例子均匀排列，类别权重仍只从AB按原公式计算。DINO、RGB注册、归一化、状态编码器、符号头、阈值、控制器和复制规则全部冻结。只改变训练数据覆盖，不同时改架构或添加人工动态规则。任何检查点选择先固定规则、仅用C tune。</p><p>这一建议尚未执行。训练前应登记逐根两侧正确、七事实、改变／未变／其他对象、UNKNOWN和隐状态误差；完整保留三个种子及旧新C结果。保留原验收记录，不通过改门槛宣布完成；若有限训练仍失败就停下报告。D、闭环、更多采集和下一种模型不包含在这次建议内。</p>')
 add('<h2>本机核验与可复查文件</h2><p>54个动作声明均先于执行落盘；36次拒绝后都有96步真实物理控制；每端[t−63,t−42,t−21,t]四个唯一帧跨2.1秒、每控制步8个物理子步。40,151个公开RGB／robot帧已逐个哈希，约2.476 GB；输入只取双相机RGB和robot_obs。分支同根复位状态最大差为0，旧训练与复制诊断的134个交付文件哈希及只读状态保持不变。独立重算了108个窗口归一化与全部1,560个方法×划分×子集×事实掩码统计。</p><p>核验日志保留了一次脚本误报：粗字符串检查把说明字段“no_private_targets…”当成私有路径；改为检查实际路径常量后通过。数据、预测、模型与冻结评估脚本没有因此改动。全部新预测只执行一次，无新优化更新。</p>')
 links=[('结果与全部逐次概率','results/evaluation.json'),('全部根汇总CSV','results/root_summary.csv'),('逐根／逐模型完整指标CSV','results/all_root_metrics.csv'),('全部逐次预测与UNKNOWN CSV','results/all_pair_predictions.csv'),('每个条件对照CSV','results/root_contrasts.csv'),('全部采集尝试CSV','results/all_collection_attempts.csv'),('54次准备调用CSV','results/preparation_attempts.csv'),('误差分层CSV','results/error_partition.csv'),('单项替换诊断','results/single_replacement_diagnostic.json'),('数据完整性核验','results/data_validation.json'),('独立最终核验','results/final_validation.json'),('冻结采集协议','configs/protocol.json'),('预测前冻结记录','configs/inference_freeze.json'),('下一次训练建议（未执行）','NEXT_TRAINING_PROPOSAL.json'),('原始根清单','../../data/predicate_state_dependence_20261014/roots_manifest.json'),('全部原始记录索引','../../data/predicate_state_dependence_20261014/data_manifest.json'),('此前一步训练报告','../state_predicates_one_step_20261014/report.html'),('此前复制效果诊断','../state_predicates_copy_effect_diagnostic_20261014/report.html')]
 add('<div class="links">'+''.join('<a href="'+href+'">'+esc(label)+'</a>'for label,href in links)+'</div><p class="muted">执行环境：本机Ubuntu 24.04.5、RTX 5080 Laptop 16GB、NVIDIA 595.91.07。仿真与特征缓存／模型预测串行占用同一个GPU锁。本报告仅交付本次授权的有限采集与冻结诊断，不宣称完整世界模型—TS/LTL指南验收完成。</p>')
 css='body{font-family:"Noto Sans CJK SC",sans-serif;color:#203449;background:#f5f7fa;max-width:1240px;margin:0 auto;padding:40px 28px;line-height:1.8}h1{font-size:34px;line-height:1.4}h2{font-size:24px;margin-top:44px}p{max-width:1120px}.lead{font-size:19px}.eyebrow,.muted,figcaption{color:#596d7e}.note{background:#e7eff6;border-left:5px solid #3a6c91;padding:18px 24px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:10px 12px;border:1px solid #dce3e9;text-align:left;vertical-align:top}th{background:#e8eef4;position:sticky;top:0}tr:nth-child(even){background:#f7f9fb}.scroll{overflow-x:auto}figure{margin:24px 0;background:white;padding:16px;border:1px solid #dce3e9}figure img{width:100%;height:auto}figcaption{font-size:14px;padding:8px 12px}.links{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.links a{background:white;border:1px solid #dce3e9;padding:10px;color:#235879}code{font-size:12px;overflow-wrap:anywhere}@media(max-width:800px){body{padding:16px}.links{grid-template-columns:1fr}h1{font-size:28px}}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态条件采集与冻结诊断</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>')
 summary={'created_utc':now(),'scope_complete':True,'roots':18,'calls':54,'preparation_calls':54,'success_changed':18,'refused_nochange':36,'uncovered':0,'C_readout_before_after_all7':[18,18],'C_full_three_seed_all7':[4,4,4],'C_copy_all7':10,'C_copy_UNKNOWN_facts':16,'C_persistence_all7':12,'C_matched_target_both_sides_full_and_copy':0,'C_matched_roots':6,'new_optimizer_updates':0,'new_D':0,'closed_loop':0,'corrections_to_model_or_rules':0,'training_proposal_only':True,'report_path':str(E/'report.html'),'figures':5,'limitations':['fixed-controller interface, not physical causality','6 C independent roots; three branches share each root','C development previously viewed','not full colour-by-family factorial','no multi-step or planning validation']};save(E/'results/summary.json',summary)
 (E/'HANDOFF.md').write_text('''# 本机有限状态条件采集与冻结诊断交付

本次授权范围已完成，未训练、未改模型/规则/阈值、未采D、未闭环、未上传。

- 项目：`/home/robot/research/projects/wm_symbolic`
- 报告：`experiments/state_predicates_state_dependence_20261014/report.html`
- 原始数据：`data/predicate_state_dependence_20261014`；18新根，AB12/C6；每根3个复位分支共54次，全部真实2.1s双端4帧。
- 54次合法准备全部成功；54次诊断为18成功改变、36拒绝不变；准备失败/原语失败/未覆盖均0，没有补采或成功配额。拒绝后96真实保持已核验。
- 54次声明执行前落盘；公开40,151帧约2.476GB已哈希。独立归一化/动作向量核验通过；134个旧训练及复制诊断交付文件未变。

## 结论

当前与真实后观测读出：AB各36/36，C各18/18，UNKNOWN0。冻结完整转移三个原种子新AB均8/36、新C均4/18。复制+冻结AB效果新AB20/36、新C10/18；C UNKNOWN16/126（持有冲突4，空手place无法绑定12）。保持当前C12/18，但所有6次成功改变都错。三种子C改变事实4/8；seed2未变95/118。

18个根的b0/b1均同动作、目标事实初值相同而真实后值不同；所有full与copy均0/18根能把两侧目标事实一起预测对，C为0/6。根内分支不是独立样本，准备还改变姿态，不能声称单因子物理因果实验。

首个C影响错误：C_statedep_close_pink_0__b0，关抽屉成功后seed2仍以0.999107预测open。所有14个C转移错误均在前后真实读出正确的情况下。单项替换为真实后继编码、同一符号头，C恢复18/18；只做离线诊断，未来未输入模型。旧AB提起均在抽屉开时、关抽屉均已有物入屉，全部成功；新条件揭露覆盖缺口。不能据此唯一证明内部因果机制。seed2新C隐状态物体MAE0.2664、场景MAE0.8439，38.19%坐标超GRU范围，多步仍未验证。

## 建议，不是已执行

NEXT_TRAINING_PROPOSAL.json：只训练既有76,224参数full转移，所有感知/读出/控制器/复制规则冻结。旧AB96+新36=132对36根；旧C24+新18=42对12根仅开发。建议一次协议原3种子×600更新=1800上限，原结构/初始化/优化器/损失形式，AB权重按同公式重算。完整报告旧新C、根配对两侧、所有七事实、UNKNOWN与隐状态。不改原验收记录，不把开发成绩当D验收，失败不扩预算。需用户后续授权才训练。

## 核验与运行入口

本次冻结协议和预测源哈希在 configs/protocol.json 与 configs/inference_freeze.json。唯一预测已完成，不应重跑 predict。
results/data_validation.json、cache_full.json、inference.json、final_validation.json均passed。final_validate.py无需GPU/不做模型forward；核验1560组统计。首次粗字符串路径检查误报保留于logs/final_validate_initial_string_match_false_positive.log，修正审核脚本检查实际路径常量后通过；没有改模型/数据/预测或冻结evaluate.py。
数据采集使用 source env.sh；缓存预测 source world_model_env.sh；单卡均走 runs/gpu.lock。采集、缓存、预测进程均已结束。实验目录名保留20261014，完成时间20261015；用量重置恢复后未重采试采，读取并沿用已有pilot_gate与pilot缓存。

可视化5张：C整体/匹配对照、全部18根热图、三类真实双相机b0/b1图和全七事实表。图轴、颜色、分母、UNKNOWN和实际意义均在图内及中文图注解释。完整CSV含逐根、逐对、准备与诊断所有尝试。
这不是完整原指南的最终验收，也未恢复更大范围自动训练。
''')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
