from audit import *
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.font_manager import FontProperties
def main():
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':11})
 fig,axs=plt.subplots(2,3,figsize=(15,10));col=['#19bb80','#e58f25','#1fa9d4'];names=['夹指族1（9＋10）','夹指族2（11＋12）','整个抽屉 link3（非前板标签）']
 sizes={}
 for s in 'AB':
  with np.load(P/f'runs/masks_{s}.npz',allow_pickle=False)as z:
   b=io.BytesIO();np.savez_compressed(b,seg_static=z['seg_static'],seg_gripper=z['seg_gripper']);sizes[s]={'mask_only_npz_compressed_bytes':len(b.getvalue()),'mask_uncompressed_bytes':int(z['seg_static'].nbytes+z['seg_gripper'].nbytes)}
 with np.load(P/'runs/masks_A.npz',allow_pickle=False)as z:
  for row,cam in enumerate(['static','gripper']):
   rgb=z['rgb_'+cam];seg=z['seg_'+cam];body=seg&((1<<24)-1);link=(seg>>24)-1;h,w=seg.shape
   masks=[(body==0)&np.isin(link,[9,10]),(body==0)&np.isin(link,[11,12]),(body==5)&(link==3)]
   maskrgb=np.ones_like(rgb)*235
   for mask,c in zip(masks,col):maskrgb[mask]=np.array(matplotlib.colors.to_rgb(c))*255
   axs[row,0].imshow(rgb,interpolation='nearest',extent=(0,w,h,0));axs[row,1].imshow(maskrgb,interpolation='nearest',extent=(0,w,h,0));axs[row,2].imshow(rgb,interpolation='nearest',extent=(0,w,h,0))
   for mask,c in zip(masks,col):
    y,x=np.where(mask)
    if len(x):axs[row,2].add_patch(Rectangle((x.min(),y.min()),x.max()+1-x.min(),y.max()+1-y.min(),fill=False,color=c,lw=2))
   for j,title in enumerate(['同期原始 RGB','解码后的可见像素类别','由可见像素极值得到的框']):
    axs[row,j].set_title(('静态相机'if cam=='static'else'腕部相机')+'｜'+title);axs[row,j].set_xlabel('x：原始图像像素');axs[row,j].set_ylabel('y：原始图像像素，向下增加')
 fig.suptitle('已有 A 场景分割实测记录——这不是当前40图，也没有重新运行仿真',fontsize=16,y=.98)
 fig.legend(handles=[Patch(facecolor=c,label=n)for c,n in zip(col,names)],loc='lower center',ncol=3,bbox_to_anchor=(.5,.065))
 fig.text(.5,.027,'灰色＝其他像素。颜色表示部件身份，非置信度。蓝框含整个抽屉可见范围，不能替代“前板＋把手”标签。\n用途：证明现有渲染链路具备 link 粒度，并直观看出抽屉标签仍需网格内语义细分。',ha='center',fontsize=11)
 fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(R/'figures/archived_sync_mask_example.png',dpi=130);plt.close(fig)
 save(E/'data/storage_evidence.json',sizes)
 s=read(E/'data/summary.json');rows=read(E/'data/source_inventory20_states.json')
 report='''# 仿真真值标签可行性：本机只读核查

可以用仿真真值生成训练／评价标签，让模型推理仍只看 RGB。本次已核实本机渲染链路能提供可见像素的 body＋link ID；两根夹指可直接分开。当前定义的“抽屉前板＋把手”还不能直接从标准 link mask 获得，因为它与腔底、侧板共用一个 drawer_link 网格。

本次没有启动新仿真、重渲染、采集、模型前向或训练，也没有把恢复标签贴到任何旧 RGB 上。40图手工全审暂存，未声称完成、未生成新的40图标签版本、未做保存预测重评。此前已封存8图用户裁决保持不变。

| 对象／问题 | 实际核实结果 | 具体含义 |
|---|---|---|
| 相机同步 | static/gripper 都在一次 getCameraImage 返回 RGB、depth、seg；适配器从同一返回 tuple 保存 pixels[4] | 不需要第二次调用相机才能获得同帧 body/link mask |
| 两根夹指 | 实际 A/B 16个源状态均记录机器人 body0，夹指族 [9,10] 与 [11,12]；分别含 finger body＋固定指尖 pad | 合并各族的可见像素，得到每根夹指可见框；不能只取9、11而漏指尖 |
| 抽屉 | 实际 table body5、drawer link3；A/B/C/D URDF 都只有整体 drawer_link，单一 visual OBJ、单个对象和材质组（123顶点、246面） | 标准 mask 分不出前板、把手、腔底和侧板；整体抽屉不等同既定标签 |
| 运行证据 | runs/masks_A.npz、masks_B.npz 都有双相机 int32 segmentation、RGB、深度、标定矩阵；本次重新读取并解码出两根夹指及指尖 | 是既有本机实测证据，非仅凭 API 常识。它们与当前40图无哈希匹配 |
| 输入隔离 | 旧适配器启用 segmentation 后会把它放进返回观察字典，而且原model_view还含深度／robot_obs | 新标签通路必须写 evaluator/annotation 私有目录；模型数据加载器明确只允许 rgb_static、rgb_gripper，不能把整个观察字典直接喂给模型 |

![已有同期分割示例](figures/archived_sync_mask_example.png)

图中绿／橙代表两个夹指族，蓝代表整个移动抽屉，灰是其他像素；颜色不代表置信度。横纵轴是原始像素坐标。右列框仅由相应可见像素的最小／最大位置得到，因此不是全物体投影框。该图展示已有 A 场景记录，不能当作当前40图的新标签。

## 当前40图可复用什么

| 图像范围 | 图数 | 实际来源 | 同期 mask | 同期深度／状态 | 目前可否直接补真值框 |
|---|---:|---|---|---|---|
| ID0–31，A/B | 32 | 官方状态初始化后，本机第96物理步的新 RGB；不是原官方 RGB | 无，采集明确 segmentation=False | 双相机深度；该步 robot_obs、scene_obs、物体／抽屉位姿；保存解析后的相机配置 | 不能直接补；恢复的像素一致性尚未验证 |
| ID32–35，C | 4 | 两条既有原生执行记录的边界帧 | 无 | RGB、robot_obs、同期物体／抽屉状态；未保存深度或同期相机矩阵 | 不能直接补；只有恢复候选信息 |
| ID36–39，D | 4 | 两条既有原生执行记录的边界帧 | 无 | 同上 | 不能直接补；只有恢复候选信息 |

逐图追溯至20个源观察文件，40/40 RGB内容哈希相同；0/40有同期mask，32/40有同期深度。完整源路径、episode根级划分及证据字段见 source_inventory.csv 和 data/source_inventory20_states.json。原分组保持16训练、16 A/B开发、4 C已见诊断、4 D已见诊断；这些不是新独立测试。

恢复限制有具体源码证据：robot_obs 记录两根手指关节位置之和，而 reset 把宽度除2后赋给两指；不能由总和保证接触时两指各自位置相同。每图也没有保存最终 camera view/projection 矩阵及所有独立机器人关节状态。A/B虽有完整 scene_obs 和同期深度，仍不能把“能复位”当成“像素精确复现”。本次没有执行恢复对比，故不宣称恢复一定失败或一定成功。

32张 A/B 的同期深度提供一条可能的离线复用路线：在相机参数／坐标约定核验后，将原深度像素回投到同期抽屉坐标，再与经确认的前板＋把手网格表面对应。这里必须额外处理深度精度、遮挡边界、面片语义和腕相机位姿，先独立验证，不能立即称真值。两指独立关节缺失仍限制夹指复原。8张 C/D 没有同期深度，条件更弱。

原官方 RGB 不在这40张中；历史官方初始 RGB 仍只能作参考，任何再仿真 mask 都只属于其同期新 RGB。即使未来某次恢复渲染看起来很像，也不得凭低平均误差直接贴标签；须严格核验对应关系、目标边界与深度／相机一致性，并明确标注恢复来源。

## 可见框、全物体投影与遮挡状态

1. **可见像素框**：对已确认的目标 mask 取 [xmin,ymin,xmax+1,ymax+1]。保留像素数、mask本身及触边信息。它会随遮挡缩小，与既定可见范围定义一致。
2. **全物体投影框**：由整个语义网格投影得到，包含被遮挡部位；世界轴对齐包围盒角点投影通常更松。应作为单独几何诊断，不能替代可见框。
3. **遮挡／出画**：零mask可能是完全遮挡、出视锥、亚像素或裁剪，不能自动解释为对象不存在。需要网格视锥关系或同状态的无干扰目标投影／遮挡对比才可细分；只知道触图像边界也不足以证明出画。未充分确定时状态仍 unknown。

## 最小后续方案（仅建议，未启动）

先固定部件语义：夹指用运行时名称映射的两个 finger＋tip 族；抽屉把现有视觉网格一次性划分成前板、把手、其余腔体面片，核对三维网格及双相机投影。不能把整体 drawer mask 冒充前板，也不能为了方便未经决定改类别定义。

采用私有标注渲染通路：保持物理 URDF、质量、碰撞和 RGB 场景不变；在冻结的同一物理时刻，按确认的面片语义执行标注用视觉副本／ID渲染，并用同刻深度验证可见面与遮挡。若视觉副本需拆成固定子link，只拆标注副本，不改实际控制环境；几何、相机、像素对齐验证未通过就不收标签。

首个有界同步检查拟保留 **A/B/C/D各1个固定状态，共4状态×2相机＝8组RGB＋depth＋mask**，不执行新技能链，不开始训练。初始化产生的物理步、渲染调用单独记账，不冒充零成本。逐状态记录所有 body/link 名称映射、每个机器人关节位置（不是只有总开口宽度）、物体／link位姿、相机矩阵、renderer、资源哈希和单调 frame/tick ID。两相机串行渲染之间不得 stepSimulation，关闭实时仿真；每个相机的RGB、深度、原生mask来自同一个返回tuple，立即复制写入同一个frame记录。额外前板语义pass在同一冻结时刻执行并校验深度。

在这8图上先核对：指尖是否包含、前板是否排除了腔底、遮挡边界是否一致、双相机是否同tick、训练加载器是否只能读取RGB。失败就保留失败样本和原因，不扩展整批采集。通过后再决定自动标签数据规模；旧40图继续作为历史可视化与错误对照，不强行用它们凑标签数量。

可以另外预注册一个明确标为“真值框输入的诊断上限对照”的下游实验：在同一批同步样本、同一根级划分和冻结下游读出上，对比预测框与真值框，定位框误差影响。它不是纯RGB系统的实际成绩，也不保证所有模型上数值必然更高；不能把真值框结果计入正式验收。本次未启动这个对照。

## 代价与当前结论

原生link mask来自现成渲染返回，主要新增解码与保存成本；额外前板语义pass才需要附加工程／渲染成本，时间尚未实测。200×200＋84×84的int32 mask每个双相机状态原始大小188224字节（约184KiB）；以下压缩量仅是两个已有探针样本，不是正式数据容量估计：

'''
 for scene,v in sizes.items():report+=f'- {scene}：仅双mask压缩NPZ {v["mask_only_npz_compressed_bytes"]}字节。\n'
 report+='''
优先级应转向可靠的同步自动标签。两指的标准link标签已有实证；前板＋把手需先补网格内语义划分；当前40图缺少同期mask，精确恢复未获证明。无需为回答这一可行性问题再训练检测器，也不应继续靠逐图猜测补齐不可靠边界。

主要源码证据：env_adapter.py:36–48（开启object/link标志与同tuple保存）；camera.py:38–44（同返回RGB/depth/seg）；play_table_env.py:184–202、225–247（串行双相机与先step后观察）；robot.py:143–165、195–202（总开口宽度和均分复位）；panda_longer_finger.urdf与四场景drawer_link.obj（部件映射）。全部绝对路径及SHA256见 configs/source_references.json。
'''
 (R/'REPORT.md').write_text(report)
 (R/'SHORT_ANSWER.md').write_text('可以借助仿真真值，监督／评价用真值、推理仍只看RGB。已从本机代码和已有分割记录核实：两指可按各自finger＋tip link族自动标注；抽屉前板／把手与腔体共享drawer_link，须先做网格内语义细分。40图全部追溯核对：0张同期mask，32张A/B有同期深度，8张C/D没有；A/B是本机第96步重渲染，不是原官方RGB。精确恢复未证明，总开口宽度不能恢复两指独立关节。建议先完成语义映射并做4状态8相机帧的同步技术检查，不整批重采、不训练。\n')
 shutil.copy2(__file__,E/'source/report.py')
 refs=read(E/'configs/source_references.json')['sha256'];assert all(sha(p)==h for p,h in refs.items())
 save(E/'data/verification.json',{'utc':now(),'all_read_source_hashes_still_match':True,'read_source_files':len(refs),'current_RGB_content_matches':40,'current_mask_count':0,'archived_masks_checked':4,'new_simulation_or_models':False,'manual40_labels_completed':False})
 files=sorted([p for root in [E,R]for p in root.rglob('*')if p.is_file()]);save(E/'configs/delivery_manifest.json',{'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}})
 print('REPORT_AND_SEAL',len(files),'files',sha(E/'configs/delivery_manifest.json'))
if __name__=='__main__':main()
