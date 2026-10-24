from common import *
import shutil,csv,re,subprocess

def table(cols,rows):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)+'\n'
def main():
 assert not(E/'configs/delivery_manifest.json').exists()
 for file in [E/'configs/inference_frozen.json',E/'configs/source_references.json',BASE/'configs/delivery_manifest.json',PREV/'configs/delivery_manifest.json']:
  for p,h in read(file)['sha256'].items():assert sha(p)==h,p
 rows=read(E/'data/states.json');metadata=read(E/'results/image_metadata.json');pred=read(E/'results/part_matches.json');context=read(E/'results/context_sensitivity.json');run=read(E/'results/inference.json');tokens=np.load(E/'results/patch_tokens.npz')['tokens'];heat=np.load(E/'results/matching_heatmaps.npz')['scores']
 assert run['images']==40 and run['queries']==56 and run['optimizer_updates']==run['simulator_calls']==run['new_downloads']==0 and run['model_state_unchanged'];assert tokens.shape==(40,256,384)and heat.shape==(2,28,256)
 assert len(rows)==14 and len(context)==4 and all(c['crop_reencoding_max_abs_diff']==0 and c['same_selected_patch_input_pixels']for c in context)
 for r in rows:
  z=np.load(r['path']);assert set(z.files)=={'rgb_static','rgb_gripper'}and sha(r['path'])==r['sha256']
 for p in pred:
  ii=next(i for i,r in enumerate(metadata)if r['index']==p['index']and r['camera']==p['camera']);score=heat[['drawer_handle','gripper_finger'].index(p['target']),ii];idx=int(score.argmax());yy,xx=divmod(idx,16);r=metadata[ii];assert p['top_cosine']==float(score[idx]);assert p['x']==(xx+.5)*r['width']/16 and p['y']==(yy+.5)*r['height']/16
 for name,rr in [('part_matches',pred),('context_sensitivity',context)]:
  with(R/f'{name}.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
 observations=[
 {'index':450,'camera':0,'target':'drawer_handle','observation':'最佳patch在抽屉把手/前板下缘附近；仅可作部件提议，未给出实例边界','interpretation':'limited same-view feasibility'},
 {'index':269,'camera':0,'target':'gripper_finger','observation':'腕部夹指参考迁移到静态图时，最佳patch落在上方柜门把手附近','interpretation':'visible cross-camera mismatch'},
 {'index':450,'camera':0,'target':'gripper_finger','observation':'最佳patch位于柜体左上缘而非当前夹指','interpretation':'visible cross-camera mismatch'},
 {'index':835,'camera':0,'target':'gripper_finger','observation':'最佳patch靠近机械臂上段/背景，不能视为夹指定位','interpretation':'part/arm confusion'},
 {'index':270,'camera':1,'target':'drawer_handle','observation':'腕部图最佳点在底部木纹处，没有辨认出抽屉把手','interpretation':'unconditional proposal despite unsupported identity'},
 {'index':1409,'camera':1,'target':'drawer_handle','observation':'最佳点落在腕部图左下木纹边角，没有辨认出抽屉把手','interpretation':'unconditional proposal despite unsupported identity'},
 {'index':1275,'camera':1,'target':'gripper_finger','observation':'最佳点靠近画面边缘的可见夹指部分，整夹爪边界与遮挡范围仍未知','interpretation':'partial visibility; not full instance localization'}]
 for o in observations:
  p=next(p for p in pred if all(p[k]==o[k]for k in ['index','camera','target']));o.update({'state_id':p['state_id'],'x':p['x'],'y':p['y'],'cosine':p['top_cosine']})
 save(E/'results/visual_review.json',{'utc':now(),'reviewer':'assistant RGB inspection, not independently annotated ground truth','formal_accuracy_not_computed':True,'observations':observations,'all14states_visualized':True})
 save(E/'results/verification.json',{'utc':now(),'passed':True,'RGB_only_files':14,'original_RGB_images':28,'inference_inputs':40,'recounted_top1_queries':56,'crop_isolation_cases':4,'frozen_weights_and_original_seals_unchanged':True,'training_download_collection':False,'independent_localization_accuracy_not_established':True})
 schema={'status':'proposed interface only; no production detector implemented','input':'current RGB image plus optional past RGB for separately validated association; never simulator mask, coordinates, robot_obs or future frames','instance_fields':{'category':'known colour block / drawer / gripper','instance_id':'assigned only by validated image association; uncertain identity remains unknown','camera':'static or wrist','localization_state':'localized / partially_localized / unknown; unknown does not mean absent','bbox':'RGB-derived tight visible box or null','visible_mask':'RGB-predicted visible parts or null; human masks allowed only as training/evaluation labels','part_visibility':'which drawer structures / gripper fingers are observed; no hidden-part completion','localization_score':'requires held-out calibration; cosine alone not confidence','feature':'same frozen DINO384 local-crop descriptor per view plus validity mask, or null','pixel_geometry':'image-normalized box/part geometry; no metric3D claim'},'node':'per-instance per-view local descriptor, visibility and type; do not silently fuse uncertain identities','edge':'ordered pair of localized node descriptors + within-camera relative pixel geometry + explicit validity; no fixed-ROI endpoint fallback','scope':'single-frame visual relation evidence; no guarantee of Holding force or In support'}
 save(E/'configs/proposed_interface.json',schema)
 txt='''# 抽屉与夹爪实例端点：有界只读可行性检查

现有DINO可以共用编码器生成局部描述，但本机现成方法尚不能可靠地产生跨画面一致的抽屉和夹爪实例。最小可行路线是先补RGB实例/部件标注，验证一个使用冻结DINO的轻量定位头，再把定位后的局部裁剪送回同一编码器。当前参考匹配只宜用于人工标注提议，不宜直接替代固定ROI后继续关系训练。

本轮已实际运行：14个既有状态的双RGB共28原图，加4个框外patch置灰输入和8个局部裁剪输入，共40张；冻结推理一次，无安装、下载、采集、训练或仿真调用。所有样本均已被使用或查看，只支持工程可行性检查。

## 本机能力核验

'''
 txt+=table(['元素','实际核验','判断'],[
 ['DINOv2-small','缓存权重成功加载；输出16×16×384 patch；推理前后参数哈希相同','可共用视觉编码器；同一坐标维度不等于跨相机语义已对齐'],
 ['既有RGB定位器','HSV最大连通域，仅红/蓝/粉三色物块','不能定位木质抽屉或白灰夹爪'],
 ['实例检测/分割','指定项目模型目录、共享模型缓存和HF/Torch缓存未找到SAM、Grounding、YOLO、OWL等权重','只报告已检查目录；没有可靠现成抽屉/夹爪定位器的证据'],
 ['其他本地权重','HULC技能执行器；MiniLM文本编码器','不把它们冒充实例检测器；未运行或改动它们'],
 ['当前独立环境','torch/torchvision/transformers可用；cv2、segment_anything、ultralytics不可用','本轮只用已装PyTorch/NumPy/Matplotlib；不安装依赖']])
 txt+='[核验目录与权重清单](../../experiments/endpoint_rgb_feasibility_20261018/results/local_inventory.json)记录的是推理前快照；[实际推理记录](../../experiments/endpoint_rgb_feasibility_20261018/results/inference.json)确认DINO已经运行。\n\n'
 txt+='## 实际探针：能提出部件位置，不能可靠给出实例\n\n选取A/B各一张已有训练RGB，人工在图像上粗框抽屉把手、单个夹指，各2个参考。框来自本助手对RGB的观察，不是独立审核的实例真值。用相应patch均值作参考，在每张图所有patch中搜索最大余弦相似位置；无固定搜索ROI、无模拟器输入、无新阈值。把手参考来自静态相机，夹指参考来自腕部；分别迁移到两种相机，共56次查询。参考图自身的匹配不计泛化证据。\n\n'
 txt+=table(['例子','匹配余弦','图像审阅与意义'],[[f"{o['state_id']} / {'静态'if o['camera']==0 else'腕部'} / {'把手'if o['target']=='drawer_handle'else'夹指'}",f"{o['cosine']:.3f}",o['observation']]for o in observations])
 txt+='余弦不是置信度，最大值总存在；当前方法没有“目标缺失”的输出，也没有实例边界。较高相似度仍可落到错误部件。静态视角多例落在抽屉前板/把手附近，但布局近似固定，尚不能排除位置先验；把手也不能代替抽屉容器结构。未用手工审阅数出一个未经独立标注支持的准确率。\n\n'
 txt+='![全部样本前半](figures/part_matches_1.png)\n\n![全部样本后半](figures/part_matches_2.png)\n\n两图覆盖全部14状态；每行左静态、右腕部。绿色圆是把手参考的最大相似patch，粉色叉是夹指参考；虚框只标A/B人工参考。标题数值为余弦，不是正确概率。跨相机出现柜门把手、机械臂与木纹误匹配；遮挡或出视野时仍返回一个点，不能作为身份一致性证明。原图像素坐标、来源和全部56结果见[匹配CSV](part_matches.csv)。\n\n'
 txt+='## 局部裁剪与全图token不是同一种信息\n\n本次将全图RGB缩放到224，使用16×16 patch。原候选把84×84腕部图填充到224，实际内容约占6×6 patch；因此本探针的特征不能直接送进旧读出头，也不是一次旧模型性能复测。\n\n'
 txt+='对4个人工参考框，先保持被池化的完整14×14 patch输入像素完全不变，再把其他patch置灰；同一冻结DINO的区域特征仍改变。另一路先取相同局部RGB裁剪，再单独编码，两个输入字节相同，输出最大绝对差均为0。\n\n'
 txt+=table(['参考部件','全图ROI余弦（改框外patch前后）','局部裁剪重新编码最大差'],[[c['template'],f"{c['full_image_ROI_cosine']:.3f}",c['crop_reencoding_max_abs_diff']]for c in context])
 txt+='![局部与上下文对照](figures/local_crop_vs_context.png)\n\n左图蓝柱的纵轴是余弦，虚线1表示完全同向；输入局部patch完全相同，差异来自其余图像对编码的影响。右图橙柱纵轴是局部裁剪编码最大差，均为0；这仅验证框外像素被隔离，不能证明框正确、框内没有背景或跨视角特征一致。置灰是强烈的合成扰动，数值不代表自然背景变化下的性能损失。\n\n'
 txt+='## 最小实施方案（本轮未训练或部署）\n\n'
 txt+=table(['环节','最小实现','需要补足／不能省略'],[
 ['RGB标注','在A/B训练来源定义并标注可见实例/部件：抽屉的活动前板及可见容器边界；夹爪两指及指尖；三色物块','记录遮挡、出框、无法判定和跨帧/跨相机身份；柜门把手、机械臂、木纹、被夹物体是困难负例。单个把手/夹指框不足以充当整个对象'],
 ['定位器','使用同一冻结DINO的全图patch，加一个小型定位/可见分割头；只训练定位头，不换编码器','需要上述监督和独立定位评估；当前两张参考匹配仅辅助标注。当前软件可实现此路线，但未验证学习效果'],
 ['节点特征','定位成功后紧致裁剪RGB，固定预处理，重新用同一冻结DINO编码为384维局部patch池化特征；有可靠可见mask时抑制框内非实例像素','所有物块和端点用同一局部路径。部分结构可保留带有效掩码的部件token；没有mask就明确框内背景仍存在。人工评估标注不得作为部署输入'],
 ['身份与视角','先保留每相机的实例特征、部件可见性和关联状态；经RGB关联验证后才融合','唯一抽屉/夹爪的类别先验不能自动证明跨画面对应；身份不确定时不合并。只许过去/当前RGB，不用未来帧或robot_obs'],
 ['缺失与遮挡','输出localized / partially_localized / unknown、置信度与有效掩码；特征缺失用null或masked token','未检测到不等于不存在；不回填固定ROI、不把零向量隐式当负例，不从仿真/上一帧强补当前观测'],
 ['关系边','有序对象对：两端的上述局部特征、同相机相对像素几何、可见性与有效掩码；节点、边读出分开但复用同一节点特征','仅是可见关系证据；不把二维接近当真实抓持/支撑。需重建新特征协议，不能拿新裁剪直接复用旧头/旧PCA结果']])
 txt+='[接口草案JSON](../../experiments/endpoint_rgb_feasibility_20261018/configs/proposed_interface.json)只定义输入输出，不冒充已实现的检测器。若不补标注、不训练定位头，也不新增外部定位能力，目前只能得到待人工确认的部件提议，不能声称已得到可靠实例级节点。人工框/掩码可用于一个明确标注的定位上限诊断，但不能混入自动推理验收。\n\n'
 txt+='## 后续测试设计与真正独立评估\n\n'
 txt+=table(['测试','比较与指标','约束'],[
 ['定位与拒绝','对RGB人工审核的可见mask/框统计IoU、部件中心误差、错误部件率；对遮挡/出框/无可辨认实例统计误检和正确拒绝','按根、相机、开合/姿态/遮挡分层；同时报告覆盖率和全部样本错误，不能靠拒绝难例提高表面准确率'],
 ['跨画面身份','同一实例的跨帧及跨相机正对，对比柜门把手/机械臂/其他物体负对；记录错误关联、身份切换、未知率','配对来自RGB审核标注，模糊样本保留未知。不能仅凭特征余弦大就算同一实例'],
 ['定位等变性与背景依赖','已知图像平移/裁切下，预测位置须随变换移动；固定局部像素改变外部背景时，比全图ROI与局部重编码','几何变换只是图像探针，不冒充物理反事实；框内背景依赖还需实例mask或有标注的匹配样本'],
 ['关系读出','定位通过后才做节点/真实对象对边的严格同结构控制，固定初始化、采样、损失与预算；FP/FN和缺失分层优先','可见框内错误与关系错误分开归因；不重启SupCon，不以本轮14图重调C阈值或决定模型采用'],
 ['独立性','先合并项目历次训练、评估、人工浏览、推理的根/episode/初始化谱系清单，并冻结排除集；本轮14状态所在根全部加入','训练仅A/B；阈值和协议在独立开发C确定后冻结；最终D选从未使用或查看的根/episode。共享初始化和相邻片段不能靠换ID变独立'],
 ['来源不足时','先只读元数据确认是否有真正未触及的来源；同时检查RGB重复与轨迹谱系','若现有来源全部已见，必须以后另行授权建立新的episode/root后才可独立评估；当前C/D不再充当新测试。本轮未新采集或预先查看新候选图像']])
 txt+='后续阈值、样本配额与成功门槛必须在新评估前书面冻结，并沿用主指南的规模与失败约束；本报告没有降低门槛。定位、关联、拒绝中任一不通过，都先作为明确能力缺口，不继续以关系训练掩盖。\n\n'
 txt+='## 交付与边界\n\n'
 txt+=f"推理设备{run['device']}，计时{run['seconds']:.3f}秒（不含准备、制图和审阅），峰值PyTorch分配{run['peak_allocated_bytes']/1024**2:.2f}MiB。56个最佳patch坐标与保存热图复算一致，4组局部裁剪隔离检查通过；原两轮封存目录逐文件校验不变。没有新增训练或下载，没有改旧模型或解除SupCon暂停。\n\n"
 txt+='本轮没有权限拒绝或环境执行阻塞。阻碍是已证实的能力缺口：现成部件匹配缺少可靠实例边界、跨画面关联和缺失判定；下一步需要RGB标注与受控的定位能力验证。完整世界模型/闭环验收并未因此完成。\n\n'
 txt+='- [实际推理记录](../../experiments/endpoint_rgb_feasibility_20261018/results/inference.json)与[验证记录](../../experiments/endpoint_rgb_feasibility_20261018/results/verification.json)。\n- [冻结探针协议](../../experiments/endpoint_rgb_feasibility_20261018/configs/protocol.json)、[人工参考来源](../../experiments/endpoint_rgb_feasibility_20261018/configs/templates.json)、[图像审阅](../../experiments/endpoint_rgb_feasibility_20261018/results/visual_review.json)。\n- [上下文数值CSV](context_sensitivity.csv)、[全部匹配CSV](part_matches.csv)、[交付哈希清单](../../experiments/endpoint_rgb_feasibility_20261018/configs/delivery_manifest.json)。\n- [前轮严格对照报告](../visual_edge_zero_context_control_20261018/REPORT.md)：不采用当前上下文边分支的结论保持。\n'
 (R/'REPORT.md').write_text(txt)
 for p in [Path(__file__),Path(__file__).parent/'render.py']:shutil.copy2(p,E/'source'/p.name)
 for stage in['prepare','probe','render']:
  log=Path(f'/tmp/endpoint_feasibility_{stage}_20261018.log');assert log.exists();shutil.copy2(log,E/'logs'/log.name)
 links=[];dest=E/'configs/delivery_manifest.json'
 for target in re.findall(r'\]\(([^)]+)\)',txt):
  p=(R/target).resolve();assert p.exists()or p==dest,target;links.append(str(p))
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'no_install_download_collection_training':True,'no_simulator_or_robot_inputs':True,'models_adopted':False,'old_manifests_unchanged':True,'figures_visually_inspected':['reference_RGB_grid.png','part_matches_1.png','part_matches_2.png','local_crop_vs_context.png'],'GPU_processes':gpu.stdout.strip().splitlines(),'report_links_checked':len(links),'remaining_capability_gap':'no reliable verified instance localizer/cross-view association/absence handling; RGB annotation and bounded localization validation needed'})
 files=sorted(p for root in[E,R]for p in root.rglob('*')if p.is_file()and p!=dest);save(dest,{'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}})
 for p,h in read(dest)['sha256'].items():assert sha(p)==h,p
 print('READONLY_ENDPOINT_FEASIBILITY_SEALED',len(files),sha(dest),flush=True)
if __name__=='__main__':main()
