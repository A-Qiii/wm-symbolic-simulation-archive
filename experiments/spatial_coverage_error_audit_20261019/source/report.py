from analyze import *
import ast,re,html,base64
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
LABELS=['左','中','右','抽屉内','滑柜左','滑柜右','持有','其他'];EN=['left','middle','right','drawer','slideL','slideR','held','other']
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def desc(value,field):
 if isinstance(value,list):return '空手'if not value else '+'.join(value)
 if field in['inside_drawer','outside_drawer']:return '在抽屉内'if value else'不在抽屉内'
 if field=='drawer_open_state':return ['关闭','半开','打开'][value]
 return LABELS[value]
def main():
 sys.addaudithook(guard);facts=read(S/'facts.json');reps=read(S/'independent_case_representatives.json');objs=read(S/'object_records.json');lookup={(r['split'],r['array_index'],r['object_index']):r for r in objs};cov=read(S/'C_coverage_records.json');nn=read(S/'C_object_feature_neighbors.json');summary=read(S/'C_coverage_summary.json')
 choices=[('common_fixed','table_left','B_coverage_old_band_04'),('common_fixed','table_middle','A_coverage_gap_03'),('common_fixed','table_right','C_core1200_01'),('common_unfixed','table_right','C_core1200_13'),('common_regression','table_right','A_coverage_old_band_03'),('common_unfixed','holding_identity_positive','C_core1200_16'),('seed_discordant_regression','holding_identity_positive','B_coverage_old_band_02'),('common_unfixed','inside_drawer','C_core1200_06'),('common_regression','inside_drawer','A_coverage_gap_02'),('common_fixed','outside_drawer','C_core1200_08')]
 selected=[]
 for category,field,rid in choices:
  q=next(r for r in reps if r['cohort']=='all_boundaries'and r['category']==category and r['field']==field and r['root_id']==rid);q=dict(q);j=q['object_index']if q['object_index']>=0 else COLORS.index(q['truth'][0]);q['focal']=lookup[q['split'],q['array_index'],j];selected.append(q)
 assert len(set(r['root_id']for r in selected))==len(selected)
 save(S/'selected_cases.json',selected)
 # Coordinates of all C h0 right objects vs actual new-training h0 same-color range.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170});fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
 for j,c in enumerate(COLORS):
  q=[r for r in cov if r['scope']=='h0'and r['truth']==2 and r['color']==c];lo=q[0]['same_color_class_x_min'];hi=q[0]['same_color_class_x_max'];ax.plot([lo,hi],[j,j],lw=12,c='#78bbc1',alpha=.45,label='New train h0 same-color right x range'if j==0 else None)
  for wrong,marker,color,label in[(True,'x','#c33d3d','Both coverage heads wrong'),(False,'o','#257a45','At least one coverage head correct')]:
   a=[r for r in q if r['both_wrong']==wrong];ax.scatter([r['x']for r in a],[j+(.07 if wrong else-.07)]*len(a),marker=marker,c=color,s=48,label=label if j==0 else None)
 ax.axvline(.1466666666667,c='black',ls='--',lw=1);ax.set(xlabel='Table-local x (metres)',yticks=range(3),yticklabels=['red objects','blue objects','pink objects'],title='C h0 right objects: 38 objects from 20 physical roots');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2);fig.savefig(S/'figures/C_x_coverage.png');plt.close(fig)
 train=[r for r in objs if r['split']=='train'and r['h0']and r['color']=='blue'and r['truth']==2];test=[r for r in cov if r['scope']=='h0'and r['color']=='blue'and r['truth']==2];focal=next(r for r in test if r['root_id']=='C_core1200_13');fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax,keys,title in[(axes[0],('x','y'),'World/table xy (metres)'),(axes[1],('static_u','static_v'),'Static-camera ROI centre (pixels)')]:
  pts=np.array([[r[k]for k in keys]for r in train]);h=ConvexHull(pts);poly=pts[h.vertices];ax.fill(poly[:,0],poly[:,1],color='#7ebdc1',alpha=.25);ax.scatter(pts[:,0],pts[:,1],c='#00868e',marker='^',label='New train blue/right h0');ax.scatter([r[keys[0]]for r in test],[r[keys[1]]for r in test],c='#999999',marker='x',label='C blue/right h0');ax.scatter(focal[keys[0]],focal[keys[1]],s=180,marker='*',c='#c33d3d',label='C13 blue: both heads wrong');ax.set(xlabel=keys[0],ylabel=keys[1],title=title)
  if keys[0]=='static_u':ax.invert_yaxis()
 axes[0].legend(fontsize=8);fig.suptitle('Inside a training convex hull is not an exact matched observation');fig.savefig(S/'figures/C13_geometry.png');plt.close(fig)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 for k,q in enumerate(selected):
  f=q['focal']
  with np.load(f['source_frame'])as z:static=Image.fromarray(z['rgb_static']);wrist=Image.fromarray(z['rgb_gripper'])
  # Overlay is a measured public ROI rectangle, not a semantic/GT segmentation or success color.
  if f['static_bbox']:ImageDraw.Draw(static).rectangle(f['static_bbox'],outline='#ffcc00',width=1)
  canvas=Image.new('RGB',(1100,470),'white');d=ImageDraw.Draw(canvas);d.text((10,10),f"{q['episode_id']} | boundary {q['boundary']} | {q['field']} | {q['category']}",fill='black',font=font);canvas.paste(static.resize((360,360),Image.Resampling.NEAREST),(10,60));canvas.paste(wrist.resize((252,252),Image.Resampling.NEAREST),(385,60));d.text((10,37),'Static RGB; yellow = measured focal ROI',fill='black',font=font);d.text((385,37),'Gripper RGB',fill='black',font=font)
  def eng(v):
   if isinstance(v,list):return '+'.join(v)or'empty'
   if isinstance(v,bool):return 'in drawer'if v else'not in drawer'
   return EN[v]
  for seed in[0,1]:
   for j,arm in enumerate(ARMS):d.multiline_text((660,65+seed*155+j*60),f"Encoder {seed} / {arm}\nPrediction: {eng(q[arm+'_seed'+str(seed)])}",font=font,fill='#666666'if arm=='control'else'#00868e',spacing=5)
  d.text((10,435),f"GT: {eng(q['truth'])}; focal {f['color']}; xy=({f['x']:.4f},{f['y']:.4f})m; ROI fraction={f['availability']:.3f}",font=font,fill='black');canvas.save(S/f'figures/case_{k:02d}.png');q['figure']=f'case_{k:02d}.png'
 save(S/'selected_cases.json',selected)
 # Context-conditioned relation coverage is descriptive; not a contact/observability proof.
 relation=[]
 for q in selected:
  f=q['focal']
  if q['split']!='externalC'or f['truth']not in[3,6]:continue
  trainrel=[r for r in objs if r['split']=='train'and r['color']==f['color']and r['truth']==f['truth']and r['valid']];near=min(trainrel,key=lambda r:np.linalg.norm(np.array([r['x'],r['y'],r['z']])-np.array([f['x'],f['y'],f['z']])));relation.append({'episode_id':q['episode_id'],'boundary':q['boundary'],'color':f['color'],'truth':f['truth'],'coordinates':[f['x'],f['y'],f['z']],'availability':f['availability'],'same_color_class_train_boundaries':len(trainrel),'same_color_class_train_roots':len(set(r['root_id']for r in trainrel)),'within_xyz_axis_ranges':all(min(r[d]for r in trainrel)<=f[d]<=max(r[d]for r in trainrel)for d in['x','y','z']),'nearest_root':near['root_id'],'nearest_boundary':near['boundary'],'nearest_xyz_m':[near['x'],near['y'],near['z']],'nearest_xyz_distance_m':float(np.linalg.norm(np.array([near['x'],near['y'],near['z']])-np.array([f['x'],f['y'],f['z']]))),'note':'All-boundary training support, not independent sample count; relation also depends on gripper/support/container geometry.'})
 save(S/'relation_coordinate_checks.json',relation)
 # Full inventory of independent-root examples for each category and field.
 flat=[]
 for q in reps:
  flat.append({k:q[k]for k in['cohort','split','root_id','episode_id','boundary','field','color','truth','category','control_seed0','coverage_seed0','control_seed1','coverage_seed1']})
 csvwrite(S/'independent_cases.csv',flat)
 names={'table_left':'桌面左','table_middle':'桌面中','table_right':'桌面右','holding_identity_positive':'持物身份（正例）','empty_hand_identity':'空手身份','inside_drawer':'抽屉内正例','outside_drawer':'抽屉外负例','drawer_open_state':'抽屉开合'}
 inventory=[]
 for field in['table_left','table_middle','table_right','holding_identity_positive','inside_drawer','outside_drawer']:
  line=[names[field]]
  for cohort in['h0','all_boundaries']:
   for cat in['common_fixed','common_unfixed','common_regression','seed_discordant_regression']:line.append(len(set(r['root_id']for r in reps if r['cohort']==cohort and r['field']==field and r['category']==cat)))
  inventory.append(line)
 lines=['# 空间覆盖实验之后：哪些错误真的改了，哪些没有','', '**结论：扩大读出训练覆盖修正了一些具体状态，但不是统一修复。初态h0没有“两个旧头都错、两个新头都对”的共同修正；动作后的部分边界有共同修正，也有共同退步。C中大量可见、坐标已被新训练覆盖的右区对象仍被两个头判成中区。**','', '本审计只读取已封存的预测、特征、公开图像和离线标签；新训练0、模型前向0、仿真0、上传0。主实验报告与12,095文件封存保持不变。','', '## 计数口径：独立根和具体状态不能混为一谈','', '“共同修正”要求两个旧头均错且两个覆盖头均对；“共同未修正”要求四个头在该事实上都错；“共同退步”要求两个旧头均对而两个覆盖头均错。“单种子/分歧退步”单独列出，不能写成两个种子共同退步。h0仅取每根b0初态；全边界每个根在每类只取最早一个案例。一个根可在不同字段或时刻同时有修正和退步，表的各列不可相加当作独立根总数。','',table(['字段','h0共同修正','h0共同未修正','h0共同退步','h0分歧退步','全边界共同修正根','全边界共同未修正根','全边界共同退步根','全边界分歧退步根'],inventory),'', 'h0没有入抽屉正例，因此其“0例”不是模型完美。持物正例没有共同修正，也没有两种子同时退步的例子；已有的是共同漏掉C16蓝块，以及个别种子的退步。全部具体根、分支、边界与四头答案列在independent_cases.csv。','', '## 十个互不重复的物理根，解释修正、未修正和退步','',table(['物理状态（根/分支/边界）','事实/真值','旧头E0→覆盖E0','旧头E1→覆盖E1','归类'],[[q['episode_id']+'/h'+str(q['boundary']),names[q['field']]+' / '+desc(q['truth'],q['field']),desc(q['control_seed0'],q['field'])+' → '+desc(q['coverage_seed0'],q['field']),desc(q['control_seed1'],q['field'])+' → '+desc(q['coverage_seed1'],q['field']),q['category']]for q in selected]),'']
 explanations=[
 '红块仍在桌面左侧。两旧头均判成中，新头均判回左；这类修正发生在动作后的视角/上下文中，不是h0的共同改善。固定E不变也能消掉这个具体错误，说明不能把所有错例都归于编码器不可用。',
 '粉块在中区、x约0.126m。两个旧头都越界判右，覆盖头都判中。这与某些其他中区对象被新头推向右区的退步并存，说明是决策边界的重新分配，不是单调增强。',
 '该C根动作后的粉块右区状态得到两个头共同修正；这证明C并非完全不可改。但同一物理根其他对象/时刻仍可错，不能拿这一帧替代C20根整体结果。',
 'C13蓝块是关键反例：x=0.18165m，距右分界约35.0mm，公共ROI可用比例1.0。它落在同颜色右区训练的xy和静态像素中心凸包内，最近训练同类x只差1.56mm，却仍被四个头判中。缺少某一段x覆盖或完全看不见，均不足以解释此例。',
 '这个蓝块x约0.255m，处于旧训练右区带附近，且两相机历史ROI均可用。旧头均正确，新头均错成中，是明确共同退步；不能把新覆盖当作只增不减的改进。',
 'C16的蓝块真实被持有，公开ROI可用比例1.0，但四个头都回答空手。新训练中最近的同色持物对象位置仅差约5.9mm，且它在同类xyz各轴范围内；仍不能把这一例简单归于坐标缺口或物体不可见。持有还要区分指间接触与外部支撑，近位置不等于严格匹配的姿态/上下文。',
 '同样是持有蓝块，E0保持正确，E1从正确退为空手。这里必须叫单种子退步；不能伪造“两种子共同退步的持物案例”。',
 'C06红块由抽屉支撑，物体可见、ROI有效；四个头都漏报InDrawer。这个同色InDrawer位置确实超出新训练xyz逐轴范围，距最近同类约27mm，因此局部覆盖不足仍是可能因素。与C13和C16相反，这例不能用已有数据排除覆盖问题；单凭这一距离也不能认定原因。',
 '新A/B空白区根中的粉块已在抽屉内；旧头均正确，新头均漏报。这是共同退步，位置覆盖干预可能改变共享读出中的关系决策。',
 '该红块实际仍在桌面中区，两个旧头却误报在抽屉内，覆盖头共同清除了误报。应将这个“抽屉外负例修正”与真正入抽屉正例识别分开，不能用它掩盖上一例的漏报。']
 for k,(q,explanation)in enumerate(zip(selected,explanations)):
  lines +=[f"### 例{k+1}：{q['episode_id']}，边界{q['boundary']}",'',explanation,'',f"![例{k+1}](figures/{q['figure']})",'图中左侧是保存的静态RGB，中央是夹爪RGB；黄色矩形是目标对象的公开ROI边框，不表示预测正确。灰字为旧头、青字为覆盖头。底部坐标单位为米，ROI fraction为四帧×双相机32个ROI token的有效比例，不是置信度。图像只做最近邻放大和边框标注。','']
 lines +=['## C的坐标与可见性检查','',table(['C h0右区对象子集','对象/独立根','同类整体x范围内','同色同类x范围内','同色同类xy凸包内','同色同类像素凸包内','静态可见/有ROI'],[[r['subset'],str(r['objects'])+'/'+str(r['roots']),r['within_pooled_x'],r['within_same_color_x'],r['within_same_color_xy_hull'],r['within_same_color_static_hull'],str(r['static_visible'])+'/'+str(r['public_roi_available'])]for r in summary if r['scope']=='h0'and r['class']==2]),'', '这里both_wrong仅指两个覆盖头同时错，与前表严格“四头皆错”的common_unfixed不同。38个右区对象来自20根，不能把38当成独立样本量。22个同时错对象来自19根，其中20个在同色同类新x范围内，6个同时位于xy和静态图像中心的训练凸包内；它们全部可见且有ROI。另一方面，多数对象并不在二维凸包内，因此“x已覆盖”也不能升级为所有位置、姿态、背景和接触组合都被覆盖。','', '![C覆盖](figures/C_x_coverage.png)','横轴为物理桌面x，三个横行是对象颜色。青色带为新训练h0同色右区的实际x范围，红叉表示两个覆盖头都错，绿圆表示至少一个正确；黑虚线是中/右分界。范围内红叉直接说明坐标范围覆盖不是充分条件。','', '![C13二维几何](figures/C13_geometry.png)','左图为xy米坐标，右图为静态相机ROI中心像素（v轴向下）。青三角是新训练蓝色右区初态，浅青多边形是其凸包，灰叉是C蓝色右区初态，红星是C13错例。凸包只说明插值区域，不表示附近存在完全相同姿态、相机上下文或其他物体布局。','']
 cc=next(r for r in cov if r['scope']=='h0'and r['root_id']=='C_core1200_13'and r['color']=='blue');nt=next(r for r in objs if r['split']=='train'and r['h0']and r['root_id']==cc['nearest_same_color_class_root']and r['color']=='blue');lines +=[f"C13蓝块xy=({cc['x']:.5f},{cc['y']:.5f})m；最近同类x的训练根{nt['root_id']}为({nt['x']:.5f},{nt['y']:.5f})m。x仅差1.56mm，但y和完整观察并不完全相同。",'',table(['C关系错例','同色关系训练边界/根','xyz各轴范围内','最近同类3D距离','公开ROI'],[[r['episode_id']+'/h'+str(r['boundary']),str(r['same_color_class_train_boundaries'])+'/'+str(r['same_color_class_train_roots']),r['within_xyz_axis_ranges'],f"{r['nearest_xyz_distance_m']*1000:.1f}mm",r['availability']]for r in relation]),'', '关系坐标表使用全部已有训练边界，仅描述支持范围，重复边界不增加独立根数。轴范围内不等于三维联合分布相同；持有/入抽屉还依赖夹爪、支撑和容器的相对关系。','', '## 能区分什么，仍不能区分什么','', '已有证据能确认：冻结E时，仅更换同容量、同预算、同标签序列的读出训练样本，就能修正部分实例并造成另一些退步。因此这些具体修正并不要求先重训E；但“猜对一部分”不能证明E对所有状态都保留了充分的因果信息。公共颜色路由和ROI几何已通过独立核验，且上面若干错例ROI完整可用，可以排除它们是路由错色或完全不可见造成的。','','C13的对象特征距离提供线索，而非结论：']
 neighborrows=[]
 for r in nn:
  if r['root_id']=='C_core1200_13'and r['color']=='blue':
   for name in['nearest_correct_class','nearest_wrong_class']:
    a=r[name];neighborrows.append([r['seed'],'同真值类最近'if name=='nearest_correct_class'else'异真值类最近',a['root_id'],LABELS[a['truth']],f"{a['cosine_distance']:.4f}"])
 lines +=['',table(['E种子','对象特征比较','训练根','真值类','余弦距离（越小越近）'],neighborrows),'', '两种子中，C13蓝块都更靠近某个中区训练对象特征。这个结果与表征的空间/场景混合相容，但不能证明坐标信息已被编码器丢弃：距离只看对象512维，实际读出还用全局上下文、颜色查询和可用比例，分类器也不是最近邻。非线性信息可能仍存在，只是当前头、优化或样本配置没有利用好。','','仍需要后续受控实验才能区分：同一冻结特征上、同训练根与标签、同预算的读出形式对照，可检验现有头的可利用性；固定新覆盖数据、读出结构与训练预算后只改变E是否在新分布训练，可检验表示适配；固定物体位置/姿态而仅改变背景或容器上下文的配对观察，可检验场景外观与关系混合。当前审计没有运行这些实验，也没有证据把所有残差归因于一个模块。','','原实验“几何准入+覆盖”的政策混合仍存在；29根旧准入通过分层只作描述，不能用本审计消除这一限制。没有增根、重训、修单例或上传。']
 md='\n'.join(lines)+'\n';(S/'REPORT.md').write_text(md);ns={'html':html,'re':re};node=next(n for n in ast.parse((O/'source/report.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='render_markdown');exec(compile(ast.Module(body=[node],type_ignores=[]),'<renderer>','exec'),ns);body=ns['render_markdown'](md)
 for p in(S/'figures').glob('*.png'):body=body.replace('figures/'+p.name,'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>只读错误分析</title><style>body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.7 sans-serif;color:#18232e}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccd6dd;padding:7px;text-align:left}th{background:#edf5f5}img{max-width:100%;height:auto}</style>'+body+'</html>');assert 'torch'not in sys.modules and 'pybullet'not in sys.modules;save(S/'report_complete.json',{'utc':now(),'selected_independent_roots':len(selected),'figures':len(list((S/'figures').glob('*.png'))),'new_forwards':0,'new_fits':0,'new_simulations':0});print('ERROR_REPORT_COMPLETE')
if __name__=='__main__':main()
