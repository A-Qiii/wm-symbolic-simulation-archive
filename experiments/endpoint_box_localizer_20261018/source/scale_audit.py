from common import *
import collections,csv,inspect,shutil
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings

def table(cols,rr):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rr)+'\n'
def main():
 rows=read(E/'data/public/images.json');aa=read(E/'data/annotations/final.json')['annotations'];sizes=[];neighbors=[]
 for r,a in zip(rows,aa):
  for cl in a['classes']:
   centers=[]
   for j,b in enumerate(cl['boxes']):
    w=b[2]-b[0];h=b[3]-b[1];pw=w/r['width']*16;ph=h/r['height']*16;cx=(b[0]+b[2])/2/r['width']*16;cy=(b[1]+b[3])/2/r['height']*16;centers.append((int(cx),int(cy)));xs=(np.arange(16)+.5)*r['width']/16;ys=(np.arange(16)+.5)*r['height']/16;inside=int(((xs>=b[0])&(xs<b[2])).sum()*((ys>=b[1])&(ys<b[3])).sum());sizes.append({k:r[k]for k in ['image_id','split','scene','camera','root_id']}|{'category':cl['category'],'box_index':j,'width_RGB_pixels':w,'height_RGB_pixels':h,'width_patch_units':pw,'height_patch_units':ph,'short_side_patch_units':min(pw,ph),'patch_centers_inside_box':inside})
   if cl['category']=='gripper_finger'and len(centers)==2:
    dist=max(abs(centers[0][k]-centers[1][k])for k in range(2));neighbors.append({k:r[k]for k in ['image_id','split','scene','camera']}|{'center_cells':centers,'Chebyshev_cell_distance':dist,'inside_same3x3_peak_neighborhood':dist<=1})
 by=collections.defaultdict(list)
 for x in sizes:by[x['split'],x['camera'],x['category']].append(x)
 groups=[]
 for (split,cam,cat),rr in by.items():groups.append({'split':split,'camera':cam,'category':cat,'boxes':len(rr),'median_width_RGB':float(np.median([r['width_RGB_pixels']for r in rr])),'median_height_RGB':float(np.median([r['height_RGB_pixels']for r in rr])),'median_short_side_patches':float(np.median([r['short_side_patch_units']for r in rr])),'min_short_side_patches':float(min(r['short_side_patch_units']for r in rr)),'boxes_short_side_less_than1patch':sum(r['short_side_patch_units']<1 for r in rr),'boxes_with_no_patch_center_inside':sum(r['patch_centers_inside_box']==0 for r in rr)})
 src=Path(inspect.getfile(Dinov2Embeddings));save(E/'results/scale_and_coverage_audit.json',{'utc':now(),'source':'frozen RGB annotations and16x16grid only; no new model inference/training','box_sizes':sizes,'groups':groups,'neighbor_pairs':neighbors,'local_DINO_resolution_code':{'path':str(src),'sha256':sha(src),'supports_positional_interpolation':True,'448_inference_or_training_run':False},'interpretation_limits':'axis-aligned box extent overestimates actual visible-part pixel occupancy; a subpatch box is not proof DINO cannot represent it; training coverage and resolution are not isolated causal factors here'})
 with(R/'box_scale_by_group.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(groups[0]));wr.writeheader();wr.writerows(groups)
 out='''# 尺度、定位头假设与训练覆盖：证据边界和下一步

本补充只使用已冻结标注和保存输出，不重跑前轮诊断、不新增推理、训练或图像。主结果及阈值保持不变。

## 小夹指与patch分辨率

200×200静态RGB缩放为224后，16×16特征网格的间距相当于原图12.5像素；84×84腕部图相当于5.25像素。下表统计的是已知可见框，不包含不确定标注。“短边<1patch”仅描述框尺度；全图注意力仍可能表示小部件，不能据此断言DINO必然无法定位。

'''
 out+=table(['集合','相机','夹指框数','宽×高中位数（原图像素）','短边中位数（patch）','短边<1patch','无patch中心落入框'],[[r['split'],'静态'if r['camera']==0 else'腕部',r['boxes'],f"{r['median_width_RGB']:.1f}×{r['median_height_RGB']:.1f}",f"{r['median_short_side_patches']:.2f}",r['boxes_short_side_less_than1patch'],r['boxes_with_no_patch_center_inside']]for r in groups if r['category']=='gripper_finger'])
 near=[r for r in neighbors if r['inside_same3x3_peak_neighborhood']];out+=f"原图小框包含的真实夹指像素少于框面积，因为轴对齐框也包含背景。全部有两根已知夹指的图中，{len(near)}/{len(neighbors)}图的两个监督中心位于彼此3×3邻域；训练中为{sum(r['split']=='train'for r in near)}/{sum(r['split']=='train'for r in neighbors)}图。中心格没有完全重合，但局部峰规则会让相邻同类中心竞争。\n\n"
 out+='可核实的是空间网格对部分静态夹指较粗，以及局部峰产生实际删除；尚未做224/448配对，不能把开发失败全部归因于分辨率。\n\n'
 out+='## 定位头的假设及已完成的单项诊断\n\n'
 out+=table(['假设','实际证据','不能推出的结论'],[
 ['一框一个16×16中心；3×3峰抑制','训练id0夹指中心score0.750、回归IoU0.973却被峰筛选删除','不能将训练漏检全归为表示不足'],
 ['局部峰可安全去重复','已完成单项去峰诊断，训练夹指TP26/27→29/30；AB开发仍5/2，种子0 FP5→6','去峰不是开发泛化修复；不采用诊断输出'],
 ['单patch384→64的头足以解码新来源','训练框拟合高；开发多数漏检的GT中心分数低于固定阈值','这可能涉及样本覆盖/特征/容量，当前未隔离各项因果'],
 ['两类共用头、无显式XY即可避免位置捷径','确实没有显式坐标、相机ID或固定ROI输入；全图DINO仍有位置与上下文','不能声称已经排除布局捷径或证明实例关系'],
 ['未输出就是可靠的缺失拒识','无not_visible样本，腕部前板全uncertain；未知前板仍大量输出','无法评价缺失拒识，更不能说已通过']])
 out+='''## 训练覆盖

训练仅4个A/B源episode、每源2状态，共16图；前板有效8框（全部静态），夹指30框。模型看到许多同源布局，训练中的额外300轮重复不增加独立样本。开发来自另外4个源episode，已被先前工作查看，但本定位头未训练这些图。训练—开发落差支持“当前小集不足以获得可靠泛化”的判断，不能单独证明扩大数量必然有效。

前板的腕部边界无法可靠判定，不能用固定框或把内部区域改叫前板来补监督。完全不可见负例计数为0；partial主要包括画面上缘截断和部分遮挡，但没有被完整遮挡的可靠缺失标签。现有coverage不能验证拒识。源episode、未知、遮挡原因和相机必须保留分层。

## 下一步最小改动建议（仅建议，本轮未执行）

1. 先检查本轮最终叠框页中前板定义与少数小夹指边界。标注一致性只是同助手重复性；仍需独立复核。腕部前板不能可靠判定就保持未知，不改变部件定义凑标签。
2. 若继续一个严格单因素定位实验，优先比较相同冻结DINO的224与448输入：448提供32×32 patch、静态网格间距降为6.25像素；保持相同样本、标注、归一化拟合规则、头结构、初始化、采样、预算与解码，把当前硬编码16的网格尺寸参数化，让两臂使用同一几何换算实现，头的参数数量不变。先测缓存/显存，再按固定两种子预算执行，不根据开发结果补图或调阈值。该建议针对已经量化的尺度问题，不保证泛化改善。
3. 本地DINO实现确实支持位置编码插值，可复用权重，不需下载大模型；448尚未实际运行。去除峰筛选是另一个因素，已在保存输出中单独诊断过，不把它偷偷并入224/448对照。不要同时更换定位头、损失和训练数据，否则失去解释性。
4. 即使更密网格改善可见定位，也不能补齐未知前板与缺失负例。可靠的缺失和独立泛化评价仍需另行冻结新的标注/来源设计；本轮不扩样。任何关系读取都应等定位与覆盖证据充分后再开始。

当前可审阅交付是标注、固定基线失败结果和上述有限证据；没有建议自动启动新的训练，原定位不足以进入关系阶段的结论保持。

[完整尺寸CSV](box_scale_by_group.csv)、[尺度与相邻中心审计JSON](../../experiments/endpoint_box_localizer_20261018/results/scale_and_coverage_audit.json)、[主报告](REPORT.md)。
'''''
 (R/'SCALE_AND_NEXT_STEP.md').write_text(out)
 with(R/'REPORT.md').open('a')as f:f.write('\n针对小夹指尺度、DINO网格、定位头假设与训练覆盖的补充证据和单因素后续建议，见[尺度与下一步](SCALE_AND_NEXT_STEP.md)。本轮未执行该建议。\n')
 shutil.copy2(__file__,E/'source/scale_audit.py');print('SCALE_AND_COVERAGE_READONLY_AUDIT_COMPLETE',json.dumps(groups,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
