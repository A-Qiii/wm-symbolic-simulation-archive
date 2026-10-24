"""Static audit figures and report; consumes only saved original/restored data."""
try:
 from official_ab_verify import *
except ModuleNotFoundError:
 from verify import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':10,'figure.dpi':145,'axes.unicode_minus':False})
def one(rows,**kw):
 out=[r for r in rows if all(r[k]==v for k,v in kw.items())];assert len(out)==1;return out[0]
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---'for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(str(v)for v in r)+' |'for r in rows)+'\n\n'
def dual(a,stage=None):
 keys=[f'{stage}_rgb_static',f'{stage}_rgb_gripper']if stage else['rgb_static','rgb_gripper']
 return np.concatenate([np.array(Image.fromarray(a[k]).resize((200,200),Image.Resampling.NEAREST))for k in keys],axis=1)
def main():
 assert read(E/'results/verification.json')['passed'];rows=read(E/'results/restoration_refined_masks.json');summary=read(E/'results/quality_summary.json');mapping=read(E/'results/color_mapping_evidence_refined.json');selection=read(E/'configs/combined_frame_selection.json');roots=selection['roots'];figs=[]
 # First uniform anchor per episode, selected before observing quality.
 fig,axs=plt.subplots(4,4,figsize=(12,11.5))
 for i,root in enumerate(sorted(roots,key=lambda x:x['root_id'])):
  row=one(rows,root_id=root['root_id'],frame_id=root['coverage_anchor_ids'][0]);a=np.load(row['source_path']);b=np.load(Path(row['folder'])/'rendered.npz');ims=[a['rgb_static'],b['t0_rgb_static'],a['rgb_gripper'],b['t0_rgb_gripper']]
  for j,im in enumerate(ims):axs[i,j].imshow(im);axs[i,j].axis('off');axs[i,j].set_title(['官方固定相机','恢复固定相机（积分前）','官方腕部相机','恢复腕部相机（积分前）'][j],fontsize=9)
  axs[i,0].set_ylabel(f"{root['scene']} episode {root['official_episode_index']}\n帧{row['frame_id']}")
  axs[i,0].text(-.03,.5,f"{root['scene']} / ep{root['official_episode_index']}\n帧{row['frame_id']}",transform=axs[i,0].transAxes,ha='right',va='center',fontsize=9)
 fig.suptitle('每个episode的第一个均匀锚点：官方原图与本机恢复图\n红、蓝、粉为物体真实颜色；画面未生成、未补帧；仅为显示缩放',fontsize=13)
 fig.tight_layout(rect=[.05,.05,1,.93]);fig.text(.5,.02,'四行是4个独立episode，列分别对照同一相机。图像接近不代表历史接触力已恢复；所有原始数组与恢复快照另存。',ha='center',fontsize=9);fig.savefig(R/'restoration_examples.png',bbox_inches='tight');plt.close(fig);figs.append('restoration_examples.png')
 # Mapping confusion and all-frame drift, report post-correction version.
 fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2,height_ratios=[1,1.2]);axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
 for ax,scene in zip(axs[:2],['A','B']):
  m=np.array(mapping[scene]['native_object_rows_original_color_columns']);norm=m/np.maximum(m.sum(1,keepdims=True),1);ax.imshow(norm,cmap='Blues',vmin=0,vmax=1);ax.set_xticks(range(3),['原图红','原图蓝','原图粉']);ax.set_yticks(range(3),['原生红物体','原生蓝物体','原生粉物体']);ax.set_title(f'{scene}场景：分割区域中的原图颜色占比')
  for(i,j),v in np.ndenumerate(norm):ax.text(j,i,f'{v:.1%}\n({m[i,j]}像素)',ha='center',va='center',fontsize=9,color='white'if v>.6 else'#222')
 rng=np.random.default_rng(170818);ax=axs[2]
 for scene,col in [('A','#3179b7'),('B','#dc8a38')]:
  for k,role in enumerate(['continuous_restoration','uniform_coverage']):
   rr=[r for r in rows if r['scene']==scene and role in r['sample_roles']];val=[1000*max(x['position_m']for x in r['one_tick_objects'].values())for r in rr];x=k+rng.uniform(-.12,.12,len(rr))+(-.15 if scene=='A'else .15);ax.scatter(x,val,s=24,alpha=.7,color=col,label=f'{scene}（每组32帧/2 episode）'if k==0 else None)
 ax.set_xticks([0,1],['连续16帧×4 episode\n检查短时恢复','均匀8锚点及后继×4 episode\n初步状态覆盖']);ax.set_ylabel('一个1/240秒物理步后的\n每帧最大物体位移（毫米）');ax.set_ylim(0,.20);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
 fig.suptitle('颜色绑定与位姿稳定性分别验证\n上图深蓝=匹配占比高；下图每点一帧，蓝=A、橙=B，横向抖动仅为显示',fontsize=13);fig.tight_layout(rect=[0,.065,1,.92]);fig.text(.5,.02,'上图为唯一红色背景掩码修正后的结果，未交换A物体槽位；初版保留。下图128帧仍只来自4 episode，不是128个独立样本。',ha='center',fontsize=9);fig.savefig(R/'mapping_and_drift.png',bbox_inches='tight');plt.close(fig);figs.append('mapping_and_drift.png')
 # Descriptive candidates, with selection rule declared and all unknown retained.
 held=sorted([r for r in rows if r['original_gripper_action']==-1 and not all(r['original_provisional_labels']['v'][:3])],key=lambda x:(x['root_id'],x['frame_id']))[0]
 inside=sorted([r for r in rows if any(r['restored_t1']['y'][3:6])],key=lambda x:(x['root_id'],x['frame_id']))[0]
 examples=[('夹爪闭合且近物体：Holding仍未知',held),('恢复后一度满足In：原始支持力仍未知',inside)]
 fig,axs=plt.subplots(2,2,figsize=(13,7),gridspec_kw={'width_ratios':[1,1.4]})
 for i,(label,r)in enumerate(examples):
  a=np.load(r['source_path']);axs[i,0].imshow(dual(a));axs[i,0].axis('off');axs[i,0].set_title(f"{r['scene']} / ep{next(x['official_episode_index']for x in roots if x['root_id']==r['root_id'])} / 帧{r['frame_id']}\n官方原图：左固定相机 / 右腕部相机",fontsize=10);ax=axs[i,1];ax.axis('off')
  cells=[]
  for j,fact in enumerate(r['original_provisional_labels']['fact_order']):cells.append([fact,'真'if r['restored_t0']['y'][j]else'假','真'if r['restored_t1']['y'][j]else'假','负例（几何）'if r['original_provisional_labels']['v'][j]else'未知'])
  tab=ax.table(cellText=cells,colLabels=['事实','恢复t0','一步后t1','原始状态暂定掩码'],cellLoc='center',loc='center',colWidths=[.37,.16,.16,.31]);tab.auto_set_font_size(False);tab.set_fontsize(8.5);tab.scale(1,1.55)
  for j in range(6):
   if not r['original_provisional_labels']['v'][j]:tab[(j+1,3)].set_facecolor('#dddddd')
   if r['restored_t1']['y'][j]:tab[(j+1,2)].set_facecolor('#ffd499')
  ax.set_title(label+'\n灰色：原始标签未知；橙色：仅恢复后的候选正例',fontsize=10)
 fig.suptitle('图像相似与夹爪闭合不能补出未保存的历史接触力',fontsize=13);fig.tight_layout(rect=[0,.07,1,.91]);fig.text(.5,.025,'各取对应候选按episode/帧编号排序的首例，仅解释局限，不参与采样或调参。t0接触力未重建；t1不是原采集时的力真值。',ha='center',fontsize=9);fig.savefig(R/'candidate_unknowns.png',bbox_inches='tight');plt.close(fig);figs.append('candidate_unknowns.png');save('report_examples',[{'label':label,'root_id':r['root_id'],'frame_id':r['frame_id'],'selection':'first lexicographic matching descriptive candidate; not used for sampling'}for label,r in examples])
 allm=one(summary,scope='all',scene='all');net=read(E/'results/network_ledger.json');verification=read(E/'results/verification.json');wrist=np.array([r['rgb_MAE_0_255']['rgb_gripper']for r in rows])
 text='# 官方A/B状态接入小试：几何可恢复，力相关正例尚不可确认\n\n'
 text+=f"已从官方ABC包有界读取中央目录、元数据及4个独立A/B episode内128帧，实际响应正文共{net['body_bytes_read']:,}字节（{net['body_bytes_read']/2**20:.2f}MiB），低于512MiB上限。全部{verification['range_requests']}个请求通过206/Content-Range/长度核验，原包ETag保持一致；128帧逐一验证CRC和SHA256。没有整包下载、对象pickle执行、C/D训练数据、特征编码或模型训练。\n\n"
 text+='**小试证明了A/B图像和几何能够较好恢复，尚未证明严格Holding/In正例标签可靠。** A原生槽位“粉、蓝、红”与原RGB一致，B为“红、蓝、粉”；恢复一次物理步的最大物体漂移为0.170毫米。128帧恢复后没有Holding正例，有50帧In候选，但原始帧不保存历史力，因此这50帧不能直接当作真实In正例。当前可保守标记几何负例，其余为未知；不据此训练。\n\n'
 text+='## 来源、隔离与抽样变更\n\n本地盘点只有debug数据，training2771帧、validation1675帧，均为D、各1个episode，因此没有作为AB来源。详见[本地盘点](../official_state_data_inventory_20261018/INVENTORY.md)。此次下载使用[官方数据入口](https://github.com/mees/calvin/blob/main/dataset/README.md)对应ABC ZIP及官方scene_info修复包。对象型npy只用pickletools反汇编检查，不调用allow_pickle=True；147个episode边界为安全数值数组。\n\n'
 text+='ABC原包和修复映射一致：B=0–598909，C=598910–1191338，A=1191339–1795044，均含端点。通用merged_config仍写D，不能套给ABC全部帧；残留config/override也不能代替按帧段确认场景。恢复只用本地已固定版本的A/B原生配置，不实例化下载配置中的代码路径。\n\n'
 text+=table(['场景/episode','官方独立episode范围（含端点）','连续片段16帧','均匀8锚点（每点另加后继）'],[[r['scene']+'/'+str(r['official_episode_index']),str(r['official_episode_inclusive']),str(r['clip_inclusive']),', '.join(map(str,r['coverage_anchor_ids']))]for r in roots])
 text+='固定seed170818先抽A/B各2个episode，未读取标签决定样本。原提案是8个均匀锚点及后继；首次执行改成连续16帧，目的是检查恢复与接触短时一致性，但当时未在提案中明确记录此变更。按后续明确指令，保留原64帧，又在查看标签前补回同4个episode的均匀锚点及后继64帧；两组无重叠，共128帧。连续组检查技术恢复，均匀组查看初步状态覆盖。两组没有新增episode，不把128帧当128个独立样本；语言窗口未被当作技能边界，人类play也未改名为固定控制器转移。\n\n'
 text+='## 图像、颜色槽位与物体尺寸\n\n![官方与恢复图像](restoration_examples.png)\n\n每行是一个episode的第一个均匀锚点，四列分别对照固定相机和腕部相机；只缩放显示，不生成或补齐画面。图中红、蓝、粉是原物体颜色。128帧的固定相机平均绝对像素误差为0.0566/255、最大单帧0.5383/255；腕部相机为'+f'{wrist.mean():.4f}/255、最大{wrist.max():.4f}/255。关节与夹爪宽度恢复误差为0，TCP位置最大差1.595毫米（记录坐标与重新计算的运动学坐标比较）。小像素误差不证明力状态一致。\n\n'
 native=[]
 for scene in ['A','B']:
  ev=read(E/'results'/f'native_evaluator_{scene}.json')
  for color,obj in ev['objects'].items():native.append([scene,color,' × '.join(f'{1000*x:.0f}'for x in obj['scaled_collision_box_m']),f"{mapping[scene]['median_color_mask_IoU'][color]:.4f}"])
 text+=table(['场景','颜色','原生碰撞尺寸（毫米）','原图颜色与恢复分割区域IoU中位数'],native)
 text+='![颜色与漂移](mapping_and_drift.png)\n\n上方矩阵的行是原生URDF物体身份，列是对应渲染分割区域里的原图颜色；深蓝表示高比例，格内给出比例与像素数。下方每点是一帧，纵轴为一步1/240秒后的最大物体位移，蓝=A、橙=B，横轴区分连续与均匀组；横向抖动仅用于显示。它同时检查绑定和位置稳定，不能把这些点作为独立成功率试验。\n\n'
 text+='A槽位按原生配置为粉/蓝/红，B为红/蓝/粉；不是按README统一R/B/P顺序，也没有照[issue #41](https://github.com/mees/calvin/issues/41)盲目全局交换。物体身份、原生尺寸与RGB区域共同支持该解释。此结论仅覆盖已验证版本和这4个episode的小样本。\n\n'
 text+='唯一一次分析修正有明确原因：A帧1785231的初始红色阈值把棕色桌面也算进去，选中23011个像素，而实际红物体只有238像素，且238个都落在正确红区；原/恢复图固定相机MAE仅0.0858/255。对全体A/B统一将红通道比例阈值从1.5改为3，其他颜色和评估阈值不动，修正后A红区IoU中位数从约0.0101变为1.0000。原始粗掩码结果及拒绝A标签的初版全部保留，没有重跑物理、改槽位、改模型、改Holding/In定义或继续搜索阈值。\n\n'
 text+='## 恢复状态不等于历史物理快照\n\n每帧重建一个新的原生世界，复位对象/机器人位置并明确假设物体初速为0；绕开普通env.reset的隐含一步积分，先performCollisionDetection记录t0，再只推进1个1/240秒物理步记录t1。每帧都是独立恢复，没有从前一帧继承速度或接触缓存。原始速度、接触力和历史电机目标没有存入帧；native robot.reset持守观察到的夹爪宽度，也不等于重现历史抓取压力。\n\n'
 text+=table(['组（仍为4个episode）','帧数','最大/中位位移 mm','超过2mm的帧','t0有承力接触的帧','t1有承力接触的帧','t1 Holding候选帧','t1 In候选帧'],[['连续短段'if s=='continuous_restoration'else'均匀锚点+后继',m['frames'],f"{m['max_object_drift_mm']:.3f} / {m['median_object_drift_mm']:.3f}",m['frames_drift_gt2mm'],m['frames_any_force_contacts_t0'],m['frames_any_force_contacts_t1'],m['Holding_restored_t1_positive_frames'],m['In_restored_t1_positive_frames']]for s in ['continuous_restoration','uniform_coverage']for m in [one(summary,scope=s,scene='all')]])
 text+='t0所有帧的接触力均为0，t1所有帧都有某处承力接触（包括桌体/地面等），但这不是Holding。t1严格双侧承力且无外部支撑的Holding候选仍为0；In候选50帧分布A18/B32，其中连续16、均匀34。数据中77/128帧的夹爪动作是闭合，不能据此填77个持物正例。没有确认原始严格Holding正例，也没有确认原始严格In支持力正例；不能把“恢复没有Held”改写成“原始样本全不持物”。\n\n'
 text+='## 原始状态的暂定已知/未知掩码\n\n保持原严格定义。对于Held，只有经过颜色绑定检查、位姿稳定且两个手指族在t0/t1都距物体大于10毫米时，才保守记几何负例。对于In，只有两时点都明确越出抽屉几何至少10毫米（或底部越出既有容差加10毫米）且位置稳定，才记负例。靠近手指、几何包含、缺少历史力或不确定状态保持未知；不把恢复力当作原始正例。2毫米稳定缓冲及上述几何余量在恢复前已记录，不是修改Holding/In语义。所有数据暂留隔离区，没有进入训练集。\n\n'
 text+=table(['组','Holding已知负事实 / 未知','In已知负事实 / 未知','原始已确认正事实'],[['连续短段'if s=='continuous_restoration'else'均匀锚点+后继'if s=='uniform_coverage'else'合计',f"{m['Holding_known_negative_fact_cells']} / {m['Holding_unknown_fact_cells']}",f"{m['In_known_negative_fact_cells']} / {m['In_unknown_fact_cells']}",'0；不可宣称正例标注通过']for s in ['continuous_restoration','uniform_coverage','all']for m in [one(summary,scope=s,scene='all')]])
 text+='分母为每帧3个颜色事实：合计每个谓词组384个事实，Holding已知负328、未知56；In已知负300、未知84。CSV中未知项的占位y=false必须与v=false一起读取，不能当负例使用。标签规模不是独立样本量。\n\n![未知候选](candidate_unknowns.png)\n\n左侧是官方原始双相机，右表对照恢复t0、一步后t1与原始状态的暂定掩码。灰色为未知，橙色仅是恢复候选正例；候选按episode/帧编号排序取首例，未用于选样或调参。它说明“看起来闭合”与“恢复后受支撑”均不能补出未保存的历史力证据。\n\n'
 text+='## 本轮结论与下一步边界\n\n数据接入、场景隔离、A颜色槽位及图像/几何恢复已有实证；严格力相关正例质量仍是阻塞。官方play可提供更多视觉状态和部分保守负例，但这轮不足以放行Holding/In正例监督。关键持物、放入、释放及切换条件仍需要按同一严格定义的自采物理证据，或经单独审查接受恢复标签作为有噪声代理；本轮未自动开展这两种后续工作，也未训练模型。\n\n'
 text+=table(['证据','内容'],[
 [f'[来源及预算]({E}/results/network_ledger.json)','全部Range、ETag、读取字节、校验状态；官方完整ZIP未下载'],
 [f'[场景映射]({E}/configs/verified_scene_ranges.json) / [最终抽样]({E}/configs/combined_frame_selection.json)','不执行pickle的核验、episode范围、抽样变更理由与冻结时间'],
 [f'[恢复协议]({E}/configs/restoration_protocol.json) / [一次掩码修正]({E}/configs/color_mask_refinement.json)','原始输入限制、t0/t1步骤、物理定义与因果证据'],
 ['[逐帧质量](frame_quality.csv) / [质量汇总](quality_summary.csv)','误差、接触变化及连续/均匀两组结果'],
 ['[逐事实标签与掩码](state_label_masks.csv)','768项Holding/In事实、已知/未知原因；训练尚未启用'],
 [f'[复核]({E}/results/verification.json) / [独立封存]({E}/configs/delivery_manifest.json)','128帧CRC/SHA、episode隔离、预算及来源代码哈希'],
 ['[端点附录](../holding_endpoint_exploration_20261018/APPENDIX.md)','前轮44文件封存不变；完整Holding诊断64文件封存不变']])
 (R/'REPORT.md').write_text(text);save('report_assets',{'figures':figs,'report':'REPORT.md','no_extra_simulation_or_model_forward':True});shutil.copy2(__file__,E/'source/report.py');print('OFFICIAL_AB_REPORT_READY',flush=True)
if __name__=='__main__':main()
