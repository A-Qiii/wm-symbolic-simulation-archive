from audit import *
import html,re,base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
from matplotlib.lines import Line2D

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def render(md):
 def inline(s):return re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',html.escape(s))
 lines=md.splitlines();out=[];i=0
 while i<len(lines):
  s=lines[i]
  if not s:i+=1;continue
  if s.startswith('|'):
   rr=[]
   while i<len(lines)and lines[i].startswith('|'):rr.append([x.strip()for x in lines[i].strip('|').split('|')]);i+=1
   out.append('<table><thead><tr>'+''.join('<th>'+inline(v)+'</th>'for v in rr[0])+'</tr></thead><tbody>');out.extend('<tr>'+''.join('<td>'+inline(v)+'</td>'for v in r)+'</tr>'for r in rr[2:]);out.append('</tbody></table>');continue
  m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',s)
  if m:out.append('<img alt="'+html.escape(m[1],quote=True)+'" src="'+m[2]+'">')
  elif s.startswith('#'):
   n=len(s)-len(s.lstrip('#'));out.append(f'<h{n}>'+inline(s[n:].strip())+f'</h{n}>')
  else:out.append('<p>'+inline(s)+'</p>')
  i+=1
 return '\n'.join(out)

def main():
 sys.addaudithook(guard)
 raw=list(csv.DictReader((D/'results/records.csv').open()));rows=[]
 for r in raw:
  z=dict(r)
  for k,v in z.items():
   if k in ['split','environment','root_id','episode_id','color','camera','initial_state']:continue
   z[k]=True if v=='True'else False if v=='False'else None if v==''else float(v)
  rows.append(z)
 calibration=read(D/'results/calibration.json');supports=list(csv.DictReader((D/'results/support_ranges.csv').open()));coverage=list(csv.DictReader((D/'results/train_coverage.csv').open()));errors=list(csv.DictReader((D/'results/right_to_middle.csv').open()))
 cpal={0:'#8B5AA3',1:'#D97919',2:'#008F91'};clabel={0:'GT table left',1:'GT table middle',2:'GT table right'}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
 def select(split,cam='static',cohort='h0',locs=(0,1,2),color='all'):
  return [r for r in rows if r['split']==split and r['camera']==cam and r['GT_location']in locs and r['visible']and(cohort=='all'or r['h0_unique'])and(color=='all'or r['color']==color)]
 # Show the physical sampling gap and the observed static projection without fitting a map.
 fig,axes=plt.subplots(2,1,figsize=(11,8.5),layout='constrained')
 groups=[('train','A','Train A, 20 roots'),('train','B','Train B, 20 roots'),('test','A','AB test A, 5 roots'),('test','B','AB test B, 5 roots'),('externalC','C','C, 20 roots')]
 for i,(split,env,label)in enumerate(groups):
  q=[r for r in select(split)if r['environment']==env]
  for loc in [0,1,2]:
   rr=[r for r in q if r['GT_location']==loc];axes[0].scatter([r['GT_x_m']for r in rr],i+np.linspace(-.12,.12,len(rr)),s=28,c=cpal[loc],alpha=.8)
 axes[0].set(yticks=range(5),yticklabels=[x[2]for x in groups],ylim=(4.5,-.5),xlabel='GT table-local x (metres; scoring/audit only)',title='Independent initial roots: actual object positions occupy narrow bands')
 markers={'train':'o','test':'^','externalC':'D'}
 for split in SPLITS:
  for loc in [0,1,2]:
   q=select(split,locs=(loc,));axes[1].scatter([r['GT_x_m']for r in q],[r['u_px']for r in q],s=32,marker=markers[split],c=cpal[loc],alpha=.65)
 axes[1].set(xlabel='GT table-local x (metres; scoring/audit only)',ylabel='Public static ROI centre u (native pixels)',title='Same static calibration: position is reflected in public image geometry')
 for ax in axes:
  ax.axvline(-.1466666667,ls='--',c='black',lw=1);ax.axvline(.1466666667,ls='--',c='black',lw=1);ax.axvspan(.087205793,.249974296,color='#6B7280',alpha=.12);ax.grid(alpha=.15)
 handles=[Line2D([],[],marker='o',ls='',c=cpal[k],label=clabel[k])for k in cpal]+[Line2D([],[],marker=markers[k],ls='',c='black',label=k)for k in SPLITS]
 fig.legend(handles=handles,loc='outside lower center',ncol=3);fig.suptitle('Physical coverage and static ROI coordinates | h0 only',fontsize=15);fig.savefig(D/'figures/physical_support.png');plt.close(fig)
 # Centre geometry: training-only hulls repeated behind fixed evaluation points.
 fig,axes=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
 stat=list(calibration['camera_signatures']['A_static'].values())[0]['calibration'];V=np.array(stat['viewMatrix']).reshape(4,4).T;Q=np.array(stat['projectionMatrix']).reshape(4,4).T
 for row,cam in enumerate(['static','gripper']):
  for col,split in enumerate(SPLITS):
   ax=axes[row,col]
   for loc in [1,2]:
    tr=select('train',cam,locs=(loc,));poly=hull([[r['u_px'],r['v_px']]for r in tr]);ax.add_patch(Polygon(poly,facecolor=cpal[loc],edgecolor=cpal[loc],alpha=.12))
    q=select(split,cam,locs=(loc,));ax.scatter([r['u_px']for r in q],[r['v_px']for r in q],s=27,c=cpal[loc],alpha=.85)
   if cam=='static':
    line=np.array([project([.1466666667,y,.46],V,Q,200,200)for y in [-.3,.2]]);ax.plot(line[:,0],line[:,1],'k--',lw=1);ax.set(xlim=(52,161),ylim=(116,86))
   else:ax.set(xlim=(0,84),ylim=(84,0))
   ax.set(xlabel='ROI centre u (native pixels)',ylabel='ROI centre v (native pixels)',title=f'{split} | {cam}');ax.grid(alpha=.15)
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color=cpal[k],label=clabel[k])for k in [1,2]]+[Line2D([],[],ls='--',color='black',label='Known physical cut projected at z=0.46 m')],loc='outside lower center',ncol=3);fig.suptitle('h0 centres: shaded regions are TRAIN-only convex hulls',fontsize=15);fig.savefig(D/'figures/centres_h0.png');plt.close(fig)
 # Box sizes cannot independently identify table zones.
 fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
 for i,cam in enumerate(['static','gripper']):
  for j,split in enumerate(SPLITS):
   ax=axes[i,j]
   for loc in [1,2]:
    q=select(split,cam,locs=(loc,));ax.scatter([r['width_px']for r in q],[r['height_px']for r in q],c=cpal[loc],s=30,alpha=.6,label=clabel[loc])
   ax.set(xlabel='ROI box width (native pixels)',ylabel='ROI box height (native pixels)',xlim=(0,30),ylim=(0,29),title=f'{split} | {cam}');ax.grid(alpha=.15)
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2);fig.suptitle('h0 box sizes: class distributions overlap',fontsize=15);fig.savefig(D/'figures/box_sizes.png');plt.close(fig)
 # Case selection: blue has most total C right->middle errors across the two seeds.
 totals={c:sum(int(r['right_to_middle'])for r in errors if r['split']=='externalC'and r['color']==c)for c in COLORS};worst=max(totals,key=totals.get);assert worst=='blue'
 case=next(r for r in rows if r['episode_id']=='C_core1200_00_b1'and r['step']==0 and r['color']==worst and r['camera']=='static')
 comparators=[]
 for loc in [1,2]:
  q=select('train',locs=(loc,),color=worst);comparators.append(min(q,key=lambda r:(r['u_px']-case['u_px'])**2+(r['v_px']-case['v_px'])**2))
 index={(x['episode_id'],x['step']):x for x in read(P/'data/core1200_v1/features/index.json')['items']}
 fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
 for ax,r in zip(axes.ravel()[:3],[case]+comparators):
  with np.load(index[r['episode_id'],int(r['step'])]['public_inputs'][-1])as z:rgb=z['rgb_static']
  ax.imshow(rgb);ax.add_patch(Rectangle((r['u_px']-r['width_px']/2,r['v_px']-r['height_px']/2),r['width_px'],r['height_px'],fill=False,edgecolor='#FFE000',lw=1.8));ax.scatter([r['projected_GT_u_px']],[r['projected_GT_v_px']],s=55,c='#63FF27',marker='+',linewidths=1.3)
  ax.set(xticks=[],yticks=[],title=f"{r['episode_id']} | blue GT {clabel[r['GT_location']].split()[-1]}\nx={r['GT_x_m']:.4f} m; ROI=({r['u_px']:.1f},{r['v_px']:.1f}) px")
 ax=axes[1,1]
 for loc in [1,2]:
  q=select('train',locs=(loc,),color='blue');ax.scatter([r['u_px']for r in q],[r['v_px']for r in q],c=cpal[loc],s=35,label=f'train blue: {clabel[loc].split()[-1]}')
 ax.scatter([case['u_px']],[case['v_px']],marker='*',s=190,c='black',label='C00 blue: both heads predict middle');ax.set(xlim=(55,154),ylim=(113,87),xlabel='Static ROI centre u (pixels)',ylabel='Static ROI centre v (pixels)',title='C error falls between training spatial bands');ax.legend(fontsize=8);ax.grid(alpha=.15);fig.savefig(D/'figures/blue_case.png');plt.close(fig)
 save(D/'results/case_selection.json',{'selection':'Color with maximum summed C right-to-middle errors across archived object seeds (descriptive, not tuning); representative previously audited C00_b1 h0; nearest same-color train h0 center per GT class only for visualization','pooled_error_counts':totals,'case':case,'nearest_train_middle_right':comparators})
 # Summaries and limitations are derived from saved audit statistics.
 def sr(split,loc,cam='static'):return next(r for r in supports if(r['split'],r['location'],r['camera'],r['cohort'],r['color'])==(split,str(loc),cam,'h0','all'))
 tr=[]
 for loc in [0,1,2]:
  r=sr('train',loc);tr.append([clabel[loc],f"{r['records']} / {r['roots']}",f"{float(r['GT_x_m_min']):.4f}–{float(r['GT_x_m_max']):.4f}",f"{r['u_px_min']}–{r['u_px_max']}",f"{r['width_px_min']}–{r['width_px_max']}",f"{r['height_px_min']}–{r['height_px_max']}"])
 cov=[]
 for cam in ['static','gripper']:
  for split in ['test','externalC']:
   for loc in [1,2]:
    r=next(x for x in coverage if(x['cohort'],x['camera'],x['color'],x['location'],x['split'])==('h0',cam,'all',str(loc),split));den=r['visible'];cov.append([cam,split,clabel[loc],den,r['training_visible'],r['inside_training_4D_axis_box']+'/'+den,r['inside_training_2D_center_convex_hull']+'/'+den,r['inside_training_GT_x_range']+'/'+den])
 percolor=[]
 for c in COLORS:
  percolor.append([c]+[next(r['records']for r in supports if(r['split'],r['camera'],r['cohort'],r['color'],r['location'])==('train','static','h0',c,str(k)))for k in [0,1,2]])
 tableq=[r for r in rows if r['camera']=='static'and r['GT_location']<3];h0q=[r for r in tableq if r['h0_unique']]
 margins=[r['center_projection_error_px']for r in h0q if r['visible']]
 extra={'generator_sha256':sha(P/'scripts/core1200_prepare.py'),'sampling_formula':'lo_x+(hi_x-lo_x)*(0.16+0.34*slot)+Uniform(-0.014,0.014); slot 0,1,2; y Uniform(-0.14,-0.05); z=0.46; random object permutation','nominal_x_bands':{e:[float(s['sampling_table_bounds_m'][0][0]+(s['sampling_table_bounds_m'][1][0]-s['sampling_table_bounds_m'][0][0])*(.16+.34*j))for j in range(3)]for e,s in calibration['scenes'].items()},'h0_static_projected_GT_vs_ROI_center_error_px':{'median':np.median(margins),'p95':np.quantile(margins,.95),'max':max(margins)},'table_label_checks':len(tableq)};save(D/'results/interpretation_evidence.json',extra)
 lines=['# 公开ROI几何与桌面中/右区：有界只读诊断','', '**结论：主要新增证据是训练空间覆盖缺口，静态相机与标签坐标没有跨场景漂移。** 现有静态ROI确实含有位置区分信息，但C右区全部在训练右区的物理x范围外，不能把读出失败唯一归因于几何信息被汇聚丢失。腕部像素几何则随相机姿态变化，存在明显类别重叠。','', '本轮只读取既有数据与已封存预测：40训练根、10个固定A/B测试根、20个固定C根；没有训练、模型前向、采集或上传。主图只取每根b0的h0，三个对象不是三个独立根；全边界结果另存CSV，不能当新增独立样本。C已被历史查看，此次是探索性诊断。','', '## 标签、物理坐标和相机核对','',table(['元素','事实','为什么重要'],[['评分边界','物理桌面局部x范围[-0.44,0.44]米；分界为-0.14666666666666667与0.14666666666666672米','A/B/C完全相同，不按各自摆放采样范围重新分三份'],['精确规则','左：x<下界；中：下界≤x<上界；右：x≥上界；距任一分界须>5毫米才有效','先有桌面支撑类别，再划分桌面区域；不是把所有物体按x分类'],['坐标变换','各场景桌面原点[0,0,0]、姿态[0,0,0]，因此本数据局部x等于世界x','不是相对相机左右或桌上三个物体的排列名次'],['已保存标签核对',f'{len(tableq)}个桌面对象边界记录，物理分界复算不一致为0','包含重复根的描述性核对'],['静态相机','本次1050个最新边界帧的完整标定完全相同；200×200、FOV 10°','A/B/C投影相同；外观纹理/柜体布局仍不同，未做因果排除'],['腕部相机','84×84、FOV 75°；视图矩阵随机械臂变化；A/B/C分别213/213/173种最新帧标定','同一u/v未对应固定桌面位置'],['已有模型输入','8维框角点、可见性、掩码占据率、视角、时间；512维对象读出没有原始框的直接旁路','深度与标定虽已保存，但未输入本轮冻结RGB/框模型；此诊断也未使用深度做预测']]),'', '实现依据是calibrated_labels.py及实际labels_ownership_v2_private.json。state_labels.py中早期SPEC的“采样范围三等分”描述没有代表本数据的最终评分规则，不能据此误判标签漂移。','','## 先看训练支持范围','',table(['GT类别','h0对象数 / 根数','真实x范围/米','静态中心u/像素','框宽/像素','框高/像素'],tr),'',table(['对象身份','训练左区数','训练中区数','训练右区数'],percolor),'','训练40根中右区只来自19个A根。中区最右约0.0872米，右区最左约0.2500米，两者之间存在约16.3厘米的训练空白；真实分界0.1467米就在空白内。即使读出保留完整静态坐标，现有训练数据也没有直接约束该空白中的决策边界。此处没有拟合或选取分类阈值。','', '根生成器core1200_prepare.py使用场景采样范围的16%、50%、84%三个位置，各加±14毫米扰动；并非覆盖整个物理类别。A三个中心约[-0.112,0.075,0.262]米，B为[-0.270,-0.100,0.070]米，C为[0.056,0.175,0.294]米。这解释了为什么标签边界不变，实际训练/测试位置却分成不同窄带。','', '![物理覆盖与投影](figures/physical_support.png)','', '图1：紫/橙/青分别表示GT左/中/右类，不表示物体颜色。上图每点为独立根h0的一个桌面物体，横轴是评分用真实x；纵向小抖动仅避免点重叠。下图横轴相同、纵轴是公开静态框中心u；圆/三角/菱形对应训练/A-B测试/C。黑虚线是固定物理边界，灰带是训练中区与右区之间的无样本区间。GT仅用于画轴和评分，没有加入模型输入。','','## 固定测试和C：重叠与外推','', 'A/B测试右区h0的5个物体有4个在训练右区x范围内；C的38个右区物体分成19个近右（0.1610–0.1886米）和19个远右（0.2806–0.3078米），38个全部在训练范围外。这个结论在全部边界仍成立：C右区466个对象记录，0个落在训练右区x范围或静态中心二维凸包内。它们来自20个根，不是466个独立测试根。','', table(['相机','集合','GT类','可见数','训练可见数','落入训练中心/尺寸四维范围','落入训练中心二维凸包','落入训练真实x范围'],cov),'', '四维范围只检查中心u/v和宽/高各自最小最大值，宽松且不表示数据密度；二维凸包是训练中心点围成的区域，落在内部也不等于有充分邻近样本，落在外部也不保证必然失败。所有训练范围与凸包只用训练集建立；表中按真实类别分组，是覆盖诊断，不能当分类器成绩。','', '![中心几何](figures/centres_h0.png)','', '图2：列依次为训练、固定A/B测试、C；行分别为静态和腕部相机。横/纵轴为原图像素u/v，v向下增大。橙点是GT中区、青点是GT右区；透明阴影始终是训练点的同类二维凸包。静态图黑虚线是已知物理分界在名义z=0.46米平面的投影，仅作几何解释，未拟合或用作预测阈值。腕部相机没有画固定分界，因为姿态会改变。','', '静态h0中，训练中区中心u=57.5–110.5像素、右区137–148.5；C中区89.5–106.5、右区116–154.5。各集合中/右的u范围均不重叠，全边界也如此：说明这些已观察样本存在直接的公开位置区分信息。但训练允许110.5到137像素之间多种未确定分界，不能在看到C后挑一个阈值，再称为跨场景泛化。','', '腕部h0的C中/右u范围重叠36像素；中区12个可见中心有10个落在C右区中心凸包内。全部边界时训练腕部中/右几何也高度重叠，且C右区仅325/466记录在腕部可见。因而相机姿态、遮挡和视角必须保留，不能用腕部框位置直接替代桌面坐标。','', '![框尺寸](figures/box_sizes.png)','', '图3：每格横轴为框宽、纵轴为框高，均为原图像素；橙/青仍代表GT中/右，按集合与相机分格。尺寸分布明显重叠，宽高单独不能作为位置区分证据。','','## 一个实际错题','', '两种子合计的C右→中错误：红223、蓝250、粉242个对象记录，因此选蓝块作代表；seed0单独蓝最严重，seed1单独粉最严重。它们是已封存预测，没有重新运行模型。','', 'C_core1200_00_b1的蓝块真实x=0.184942米，距右区边界38.28毫米，远离5毫米模糊带。静态ROI中心(123.0,103.5)像素、框18×17、227个可见像素；标定投影的GT中心(123.006,103.666)，差0.166像素。腕部也有67个可见像素。两个对象头都判成中区。','', '按静态二维中心距离选择的同色训练参照：B18中区蓝块x=0.0698米、框中心(107.0,92.5)；A11右区蓝块x=0.2652米、框中心(141.0,103.5)。C错题在训练位置带之间，框本身未丢失或错位。该例支持空间外推困难，不能据此证明外推是唯一因果机制。','', '![蓝块实际案例](figures/blue_case.png)','', '图4前三格都是保存的原静态RGB；黄框是公开分割得到的蓝块ROI，绿十字是仅供审计的GT中心投影。右下橙/青点为训练中/右区蓝块中心，黑星为C错题；未拟合边界。训练参照按距离选取，不根据成功率筛选。','','## 一个后续建议与明确边界','', '**优先做一次训练布局覆盖对照，保持对象路由、模型结构和训练预算固定。** 将A/B训练摆放从三个窄带改为按物理左/中/右区域分层的连续采样，覆盖当前中/右边界两侧及现有右区训练范围外的位置；保留边界有效性与物体间距约束，并打散颜色与位置关系。先冻结采样方案，再用同等根数/转移预算比较，C不能参与训练或阈值选择。若需要新的盲测结论，应另外冻结未查看的新根；当前C只能作探索性复查。','', '这是针对已证实的覆盖缺口的一项修改建议。本轮没有拟合线性探针，没有新增几何旁路或深度输入，没有修改阈值、采集或训练。现有静态ROI可分性不能证明冻结神经表示保留了这些信息，也不能排除纹理、布局上下文或读出映射的影响；本诊断不承担动力学自由预测或完整指南验收。','','## 文件与复核','', 'results/records.csv包含6300行对象×相机记录；support_ranges.csv包含训练及固定评价的逐类/逐色范围；train_coverage.csv与class_overlap.csv记录重叠和外推；right_to_middle.csv是封存预测的错误分解。calibration.json保存精确标定与标签审计，case_selection.json记录选例，interpretation_evidence.json记录采样公式来源。','', '仅使用NumPy/YAML和Matplotlib做CPU计算，实际几何统计约3.54秒；10,000级以下记录，没有模型前向。442个旧封存文件及清单核验未变。图表说明颜色、轴、条件与局限；文件完整性与最终SHA清单见final_verification.json、manifest.json。','']
 md='\n'.join(lines);(D/'REPORT.md').write_text(md);body=render(md)
 for p in(D/'figures').glob('*.png'):body=body.replace('figures/'+p.name,'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
 (D/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>ROI空间几何诊断</title><style>body{max-width:1160px;margin:36px auto;padding:0 24px;font:17px/1.7 sans-serif;color:#18212b}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}td,th{border:1px solid #d0d7de;padding:7px;text-align:left}th{background:#eef4f4}img{max-width:100%;height:auto}h1,h2{line-height:1.35;margin-top:1.7em}strong{color:#006d77}</style>'+body+'</html>')
 assert protected()==read(D/'configs/protected_before.json');print('REPORT_COMPLETE',D/'REPORT.html',flush=True)
if __name__=='__main__':main()
