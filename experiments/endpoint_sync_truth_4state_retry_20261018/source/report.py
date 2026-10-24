from common import *
import csv,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
def main():
 results=read(E/'results/results.json');v=read(E/'results/independent_verification.json');summary=read(E/'results/collection_summary.json')
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':12})
 colors={'wood':'#ed9d28','handle':'#c357b2','other':'#a5b1bc','finger1':'#20a16a','finger2':'#367bd4','unknown':'#e3312b'}
 fig,axs=plt.subplots(4,4,figsize=(19,17));metricrows=[]
 for row,state in enumerate(results):
  s=state['spec']['scene'];d=E/'data'/s/'annotation_private'
  for ci,cam in enumerate(['static','gripper']):
   with np.load(d/f'primary_{cam}.npz',allow_pickle=False)as z:rgb=z['rgb']
   with np.load(d/f'semantics_{cam}.npz',allow_pickle=False)as z:q={k:z[k]for k in z.files}
   lab=read(d/f'visible_labels_{cam}.json');sem=q['drawer_parts'];h,w=sem.shape;display=np.full_like(rgb,246)
   for key,m in [('wood',sem==1),('handle',sem==2),('other',sem==3),('finger1',q['finger1']),('finger2',q['finger2']),('unknown',sem==255)]:display[m]=np.asarray(matplotlib.colors.to_rgb(colors[key]))*255
   ax0,ax1=axs[row,ci*2:ci*2+2];ax0.imshow(rgb,interpolation='nearest',extent=(0,w,h,0));ax1.imshow(display,interpolation='nearest',extent=(0,w,h,0))
   for name,c in [('finger1',colors['finger1']),('finger2',colors['finger2']),('drawer_front_plus_handle',colors['wood'])]:
    b=lab[name]['visible_bbox_xyxy']
    if b:ax0.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=c,lw=1.7,linestyle='-'if lab[name]['bbox_valid']else'--'))
   ys,xs=np.where(sem==255)
   for y,x in zip(ys,xs):
    ax1.annotate(f'未知 ({x},{y})',xy=(x+.5,y+.5),xytext=(w*.45,h*.30),color=colors['unknown'],fontsize=11,arrowprops={'arrowstyle':'->','color':colors['unknown'],'lw':1.4})
   cn='静态'if cam=='static'else'腕部';ax0.set_title(f'{s}｜{cn} RGB＋可见框',fontsize=14);ax1.set_title(f'{s}｜{cn} 语义mask',fontsize=14)
   for ax in [ax0,ax1]:ax.set_xticks([0,w//2,w]);ax.set_yticks([0,h//2,h]);ax.tick_params(labelsize=10);ax.set_aspect('equal')
   valid=lab['drawer_front_plus_handle']['bbox_valid'];ax1.set_xlabel('前板框有效'if valid else'前板框无效：保留未知',color='#23834b'if valid else'#b13831',fontsize=12)
   st=state['summaries'][ci]['semantic'];metricrows.append({'scene':s,'camera':cam,'drawer_initial_q_m':state['spec']['drawer_initial_q_m'],'native_drawer_pixels':st['native_drawer_pixels'],'assigned_pixels':st['assigned_pixels'],'unknown_pixels':st['unknown_pixels'],'front_pixels':lab['drawer_front_plus_handle']['visible_pixels'],'front_bbox_valid':valid,'front_bbox_xyxy':lab['drawer_front_plus_handle']['visible_bbox_xyxy'],'finger1_bbox':lab['finger1']['visible_bbox_xyxy'],'finger2_bbox':lab['finger2']['visible_bbox_xyxy'],'max_all_candidate_depth_error_mm':st['depth_error_max_m']*1000,'RGB_depth_max_difference':0})
 fig.suptitle('4状态 × 2相机同步技术样本｜同步及RGB／深度一致性 8/8｜前板完整细分 6/8',fontsize=21,y=.985)
 legend=[Patch(facecolor=colors[k],label=l)for k,l in [('wood','前板外面／边缘'),('handle','把手'),('other','排除的腔体／侧板／内面'),('finger1','夹指族1'),('finger2','夹指族2'),('unknown','未知像素')]]
 fig.legend(handles=legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.055),fontsize=12)
 fig.text(.5,.028,'每行一个固定状态，抽屉初始位移 A=0、B=0.10、C=0.22、D=0.275米。横纵坐标均为原始像素；静态200×200，腕部84×84。\n实线框通过本轮完整性规则；虚线框仅诊断，不作有效监督。颜色表示部件身份，非置信度。两处红色未知保留，门槛未放宽。\n这些是新同步技术样本，不是旧40图；重复姿态和小样本仅验证标签管线，不证明检测泛化。',ha='center',va='center',fontsize=11)
 fig.tight_layout(rect=(0,.13,1,.96),h_pad=1.5,w_pad=1.0);fig.savefig(R/'figures/sync_truth_8views.png',dpi=130);plt.close(fig)
 # Two preserved outliers, shown with original pixel coordinates and no repair.
 fig,ax=plt.subplots(2,2,figsize=(11,9))
 for row,(s,x,y)in enumerate([('B',157,134),('D',128,188)]):
  d=E/'data'/s/'annotation_private'
  with np.load(d/'primary_static.npz',allow_pickle=False)as z:rgb=z['rgb'];depth=z['depth_m']
  with np.load(d/'semantics_static.npz',allow_pickle=False)as z:sem=z['drawer_parts'];err=z['depth_error_m']
  sl=(slice(y-4,y+5),slice(x-4,x+5));extent=(x-4,x+5,y+5,y-4)
  ax[row,0].imshow(rgb[sl],interpolation='nearest',extent=extent);ax[row,0].add_patch(Rectangle((x,y),1,1,fill=False,color='#ff2222',lw=2));ax[row,0].set_title(f'{s}静态原RGB｜红框=保留未知像素')
  im=ax[row,1].imshow(depth[sl]*1000,interpolation='nearest',extent=extent,cmap='viridis');ax[row,1].add_patch(Rectangle((x,y),1,1,fill=False,color='#ff2222',lw=2));ax[row,1].set_title(f'同期深度｜该像素误差{err[y,x]*1000:.2f}mm');fig.colorbar(im,ax=ax[row,1],fraction=.046,pad=.04,label='相机轴向深度（mm）')
  for aa in ax[row]:aa.set_xlabel('原始像素 x');aa.set_ylabel('原始像素 y');aa.set_xticks([x-4,x,x+4]);aa.set_yticks([y-4,y,y+4])
 fig.suptitle('两处未知位于局部深度跳变附近｜图像和标签均未修补',fontsize=15);fig.tight_layout(rect=(0,.06,1,.94));fig.text(.5,.02,'深度色条由紫到黄表示由近到远；不是语义或置信度。边界差异是可疑原因，本轮未另渲染确认，因此保留unknown。',ha='center',fontsize=11);fig.savefig(R/'figures/unknown_depth_pixels.png',dpi=130);plt.close(fig)
 with(R/'per_view_metrics.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(metricrows[0]));w.writeheader();w.writerows(metricrows)
 primary_bytes=sum(p.stat().st_size for p in(E/'data').rglob('primary_*.npz'));sem_bytes=sum(p.stat().st_size for p in(E/'data').rglob('semantics_*.npz'));data_bytes=sum(p.stat().st_size for p in(E/'data').rglob('*')if p.is_file())
 storage={'primary8_RGB_depth_buffer_depth_m_native_mask_compressed_bytes':primary_bytes,'semantic8_compressed_bytes':sem_bytes,'all_data_with_eager_and_baseline_duplicates_snapshots_configs_bytes':data_bytes,'description':'This includes duplicate integrity captures and snapshots. Do not treat full audit bytes as unique training storage. No extrapolation to a formal collection.'};save(E/'results/storage_summary.json',storage)
 lines=['|场景／相机|抽屉像素|已核验|未知|前板框|','|---|---:|---:|---:|---|']
 for row in metricrows:lines.append(f'|{row["scene"]}／{row["camera"]}|{row["native_drawer_pixels"]}|{row["assigned_pixels"]}|{row["unknown_pixels"]}|'+('有效'if row['front_bbox_valid']else'无效，仅诊断')+'|')
 report=f'''# 同步仿真真值技术验证：4状态8图已保存，前板完整细分6图通过

原定A/B/C/D各一次修复后验证已完成，8组原始RGB、同期深度和原生object/link mask都已可靠保存并重新读取。4状态在所有相机调用前后物理快照完全一致；8图启用／关闭link-ID输出的RGB及深度缓冲逐像素完全相同，最大差0，双相机间没有物理推进。**这证明同步与保存通路可用；严格前板完整细分门槛为6/8，不能宣称8/8全通过。**

此前A状态的保存失败封存在 endpoint_sync_truth_4state_20261018，未改动；本轮经明确授权仅修复二进制元数据序列化及原始缓冲优先保存，不改选择、语义或质量门槛。共发生旧失败1次＋本轮成功4次world实例化；可用技术状态仍是原定4个，不是5个独立样本。

## 实际标签结果

{chr(10).join(lines)}

共{v['native_drawer_pixels']}个原生抽屉像素，{v['independently_verified_assigned_pixels']}个面片归属经独立平面交点／重心坐标方法复核，2个保持unknown。两指依据运行时各自finger＋tip族合并，共16个有效可见框。前板＋把手有6个有效可见框；B静态、D静态各有一个未知抽屉像素，依事前“全部抽屉像素完成分类才给有效前板框”规则，对应2个前板框无效。它们的正像素mask和诊断bbox保留，但不能作为完整监督标签。

![8组同步技术样本](figures/sync_truth_8views.png)

橙色为前板外面／露出边，紫色为把手，灰色为排除的腔体／侧板／内面，绿／蓝为两个夹指族，红色为未知。实线框是有效可见框，虚线仅用于诊断。每行同一固定场景状态，静态及腕部相机都在同一物理tick采样；坐标为各自原始像素，分辨率分别200×200及84×84。颜色不是概率或置信度。

## 两处未知与未放宽的边界

B静态(x=157,y=134)的最近网格射线深度与原生渲染深度相差107.207mm；D静态(128,188)相差21.134mm，均远超冻结的0.5mm门槛。两处都在局部深度跳变附近：图形光栅覆盖和精确像素中心射线在表面边界处的差异是可疑原因，但本轮没有额外渲染形成独立因果证明，因此不将推测写成已证实结论，也不调整射线、像素或阈值去消掉它们。

![保留的未知像素](figures/unknown_depth_pixels.png)

左侧是原RGB局部，右侧是同帧深度；红框准确圈出一个原始像素，紫到黄的色条表示相机轴向深度由近到远。该图解释未知来自深度一致性检查，而非模型输出差或人工不喜欢某个框。

## 语义与同步如何成立

四场景的抽屉视觉网格几何一致，面片索引固定：44个前板外面／露出边、178个把手、24个排除的腔体／侧板／内面。为保持原定义“排除interior”，内朝向腔体的前板面82/83归入排除类。没有把整个drawer_link当作前板；A闭合视角中两者可见像素相同，只是当时腔体未露出。B/C/D打开视角已展示分离。

每个相机一次getCameraImage返回RGB、深度和原生mask，回调立即将三者原子写入NPZ并用allow_pickle=False重新打开比较。两相机及后续flag-off重复检查之间不推进物理。对drawer_link可见像素，使用原相机矩阵、固定像素中心、原视觉网格缩放和同期link变换求最近面片，再与同期深度交叉核对；没有额外语义渲染，没有增添或改变body、材质、颜色、碰撞、质量或相机外观。

所有原生bytes统一保存带编码标签的base64，并仅在严格UTF8解码成功时附文本字段，原始字节始终保留；448个bytes字段往返核验通过，其中4个不能解码为UTF8而以原始字节保留，不猜测为资源路径。资源身份另外由原URDF/OBJ路径及SHA256确定。原始link名称用于映射时严格UTF8解码，未做猜测或替换。

## 检查、耗时与大小

- 8组RGB／深度flag-on对flag-off：全部逐像素一致；物理状态与关节、速度、link位姿快照前后一致。
- 48个NPZ文件全部无pickle读取成功，包括即时缓冲、副本、语义与几何；公开输入每状态只含rgb_static、rgb_gripper，真值在annotation_private目录。
- 独立复核27664个已分配抽屉像素的面片类别、三角形内重心坐标及≤0.5mm深度误差；框与对应mask像素极值一致。
- 修复后4状态批次计时{summary['elapsed_seconds_including_close']:.3f}秒，范围是模块导入后到全部环境关闭，含初始化、渲染、落盘和语义计算；初始化共8物理步，首图后0步。没有技能调用或训练。
- 8个主RGB／深度缓冲／米制深度／ID mask压缩包合计{primary_bytes:,}字节；8个语义包{sem_bytes:,}字节；含重复完整性捕获、快照和配置的全部data为{data_bytes:,}字节。不能把这些重复审计副本当成正式数据存储估算。

每状态保留完整body/link名称映射、各关节位置与速度、物体和link位姿、原生序列化、相机矩阵／分辨率／near/far、渲染标志、配置、资源哈希以及frame/tick/call索引，可追溯与复查。门槛、原始源码及技术重试补充见configs；逐相机表见per_view_metrics.csv。

## 能证明什么、接下来还缺什么

现在可以用该同步通路生成两指的可见像素标签，也能在多数本次视图中可靠区分前板＋把手与整体抽屉。无需再靠助手逐图猜测明显部件边界。严格的全部前板完整细分仍有2/8未通过，所以不能直接宣布批量自动标签已验收或把所有mask当精确金标准。

可见bbox仅包住实际可见像素，没有补全被遮挡部分。触边单独记录，不直接推断出画；遮挡和出画原因本轮没有独立几何测试，仍unknown。D腕部把手零像素也只表示未观察到其像素，不等于把手不存在。

这4状态具有相同初始机器人姿态与很小覆盖，不能支持检测器泛化结论，更不是独立测试。8图是新的同步技术样本，绝不将其mask贴回旧40图。若后续扩大，应先决定如何处理边界unknown与监督有效性，并固定根级分组和覆盖计划；本轮到4状态结束，未训练、未跑SupCon、未扩大采集。旧标签和模型保持原封存。
'''
 (R/'REPORT.md').write_text(report);shutil.copy2(__file__,E/'source/report.py')
 print('REPORT',primary_bytes,sem_bytes,data_bytes)
if __name__=='__main__':main()
