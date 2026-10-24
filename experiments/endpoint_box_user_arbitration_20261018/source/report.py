from common import *
import os,shutil,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from PIL import Image

def rel(p):return os.path.relpath(p,R)
def link(p,s):return f'[{s}]({rel(p)})'
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(str(v).replace('|','/')for v in r)+' |'for r in rs])
def main():
 assert not(R/'REPORT.md').exists();check_manifest(E/'configs/user_labels_and_audit_frozen.json');check_manifest(E/'configs/evaluation_frozen.json');assert read(E/'results/verification.json')['passed']
 data=read(E/'results/evaluation.json');ann={a['image_id']:a for a in read(E/'data/labels/user_semantics_v2.json')['annotations']};audit=read(E/'data/audit/AB_readonly_findings.json');dec=read(E/'data/labels/user_decision_ledger.json');meta={r['image_id']:r for r in read(E/'data/images_selected8.json')};fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
 def raw(i):return np.asarray(Image.open(E/f'data/observations/id{i:02d}.png'))
 def rect(ax,b,color,ls='-',lw=1.8):
  x,y,x1,y1=b;ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor=color,linestyle=ls,linewidth=lw))
 # Four cases retain the panel order in the user-facing arbitration image.
 cases=[(0,'gripper_finger',(25,35,105,82),'左上：恢复用户确认的两根夹指','原橙色两框 → 本图紫色实框；计分'),(8,'gripper_finger',(40,95,110,145),'右上：水平白段是把手','另一根指的可见边界未定；整类unknown'),(1,'drawer_front',(0,0,84,84),'左下：棕色木板属于会动的抽屉','前板与其它木面边界未定；前板unknown'),(9,'drawer_front',(0,0,84,84),'右下：斜木边是前板','绿框含可见前板与把手；partial，计分')]
 fig,axs=plt.subplots(2,2,figsize=(13,11));fig.subplots_adjust(left=.06,right=.98,bottom=.09,top=.86,hspace=.42,wspace=.18)
 for ax,(i,c,crop,title,note)in zip(axs.flat,cases):
  ax.imshow(raw(i),interpolation='nearest');a=next(ca for ca in ann[i]['classes']if ca['category']==c)
  for b in a['boxes']:rect(ax,b,'#a42bbe'if c=='gripper_finger'else'#129b70')
  if i==8:
   rect(ax,[55,120,81,133],'#c2442c','--');rect(ax,[88,114,98,130],'#a42bbe',':');ax.text(.02,.03,'红虚框仅指示把手位置，\n不是整块干净负样本',transform=ax.transAxes,fontsize=9,color='#9b301e',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
  if i==1:rect(ax,[25,58,63,75],'#777777',':')
  x,y,x1,y1=crop;ax.set_xlim(x-.5,x1-.5);ax.set_ylim(y1-.5,y-.5);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素');ax.set_title(title+'\n'+note,fontsize=10,pad=12)
 fig.suptitle('用户语义裁决后的新版本：两项纳入计分，两项保留边界未知\n紫实框＝计分夹指；绿实框＝计分前板；点框＝仅供辨认、未计分；不含模型预测',fontsize=13);fig.savefig(R/'figures/user_adjudication4.png',dpi=160);plt.close(fig)
 # Context explanation uses the same raw pixels, with no model-generated completion.
 fig,axs=plt.subplots(1,3,figsize=(15,5))
 for ax,crop,title in zip(axs,[(0,0,200,200),(55,119,83,134),(35,80,120,150)],['完整原图：部件与场景位置','紧致外观裁切：连接信息被裁掉','扩大局部：保留掌部、蓝块及邻近灰指']):
  ax.imshow(raw(8),interpolation='nearest');x,y,x1,y1=crop;ax.set_xlim(x-.5,x1-.5);ax.set_ylim(y1-.5,y-.5);rect(ax,[55,120,81,133],'#c2442c','--');ax.set_title(title,fontsize=10);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素')
 fig.suptitle('同一图像、同一白段：用户确认为把手；紧裁可减少背景，也会删除身份判断所需上下文\n红虚框仅定位用户所指白段；右端跨邻近部件，不能整框直接作训练硬负例',fontsize=12);fig.tight_layout(rect=(0,0,1,.87));fig.savefig(R/'figures/context_vs_tight_crop.png',dpi=160);plt.close(fig)
 # Representative read-only issues are chosen by visual audit before model-file reads.
 lookup={r['image_id']:r for r in audit};fig,axs=plt.subplots(2,2,figsize=(12,10))
 for ax,i,title,crop in zip(axs.flat,[13,21,28,30],['训练腕部：确认旧框过宽并跨入黑把手','开发腕部：确认右界跨入背景把手','开发静态：连接边界疑似，未改标注','开发静态：遮挡/可见长度待核验'],[(0,0,84,40),(0,0,84,40),(20,40,90,95),(80,65,155,115)]):
  ax.imshow(raw(i),interpolation='nearest')
  for b in lookup[i]['original_boxes']:rect(ax,b,'#a42bbe')
  x,y,x1,y1=crop;ax.set_xlim(x-.5,x1-.5);ax.set_ylim(y1-.5,y-.5);ax.set_title(f'id{i:02d} | '+title,fontsize=10);ax.set_xlabel('x 原图像素');ax.set_ylabel('y 原图像素')
 fig.suptitle('只读审查示例：紫框全部是原封存夹指标注，无模型输出、无自动新框\n“含有背景”不必然错；需看该边界是否由真实指体的可见端点支持',fontsize=12);fig.tight_layout(rect=(0,0,1,.91),h_pad=3);fig.savefig(R/'figures/readonly_audit_examples.png',dpi=160);plt.close(fig)
 def g(res,seed,var,ver,support,split='selected8_all',cat='all_categories'):
  return next(x for x in data['groups']if all(x[k]==v for k,v in {'resolution':res,'seed':seed,'variant':var,'label_version':ver,'support':support,'split':split,'category':cat,'camera':'all'}.items()))
 cfg=[(224,0),(224,1),(448,0),(448,1)];fig,axs=plt.subplots(1,3,figsize=(15,5));xx=np.arange(4)
 for ax,key,desc in zip(axs,['TP','FP','FN'],['匹配参考框 TP','未匹配输出 FP','漏检参考框 FN']):
  common=[g(r,s,'corrected','user_v2','common_v1_v2')[key]for r,s in cfg];added=[g(r,s,'corrected','user_v2','added_v2_vs_v1')[key]for r,s in cfg]
  ax.bar(xx,common,color='#808e9b',label='与v1共同范围：18参考框');ax.bar(xx,added,bottom=common,color='#e39a31',label='本次新纳入：3参考框')
  for j,(a,b)in enumerate(zip(common,added)):
   ax.text(j,a/2,str(a),ha='center',va='center',color='white',fontsize=11)
   if b:ax.text(j,a+b/2,'+'+str(b),ha='center',va='center',color='black',fontsize=10)
   ax.text(j,a+b+.3,str(a+b),ha='center',fontsize=10)
  ax.set_ylim(0,22);ax.set_xticks(xx,[f'{r}/s{s}'for r,s in cfg]);ax.set_xlabel('固定模型分辨率 / 种子');ax.set_ylabel('框数');ax.set_title(desc);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0].legend(fontsize=8,loc='upper left');fig.suptitle('保存的修正解码输出：新增计分范围的贡献（模型未改变）\n灰色共同部分与v1完全相同；橙色来自恢复两根夹指及新增一块前板，总参考框18→21',fontsize=12);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(R/'figures/saved_prediction_decomposition.png',dpi=160);plt.close(fig)
 audit_csv=R/'AB_readonly_findings.csv'
 with audit_csv.open('w')as f:w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
 lines=['# 用户语义裁决后的标签版本与保存预测重评\n']
 lines.append('本轮已逐项接受用户对四幅图的部件辨认，并另存新版本。左上两根夹指和右下斜前板已有可用可见框，恢复/新增计分；右上白段明确是把手，但另一根夹指的可见边界仍不稳，左下移动木板也尚不能与腔底/其它抽屉木面精确分开，二者保留unknown。**语义已确认不等于用户完成逐像素标注**。本轮没有训练、前向、特征提取、阈值变化或新解码。')
 lines.append('用户原话来源：2026-10-18 12:02:56 UTC，消息 `Sentinel_e8ac4202db30819186484f78c1871a42`；对应附件 Library ID `libfile_057f412211848191935b3d5681bc0795`。可信对话由父线程转交，原话保存在'+link(E/'data/evidence/user_transcript.json','用户证据记录')+'。原图SHA和颜色映射已核对。')
 lines.append('## 四个面板的对应与处理\n')
 lines.append(table(['用户看到的位置','原图 / 类别','接受的用户语义','新版本处理'],[['左上','id0 / 夹指','橙色框的两个才是夹指','恢复[65,49,77,65]、[78,48,91,67]；partial；不是蓝色水平候选'],['右上','id8 / 夹指','水平白段是抽屉把手','排除该白段作为夹指；右侧真实灰指仍可见，另一根边界未定，整类unknown'],['左下','id1 / 前板连把手','所问棕色木板属于会动的抽屉','接受父对象/运动语义；前板与腔底分界未定，仍unknown，不以把手框代替前板'],['右下','id9 / 前板连把手','斜木边就是抽屉前板','新增[0,47,84,84]；partial/out_of_frame，不补画外部分']]))
 lines.append('原裁决图橙色虚线来自原标注、蓝色点线来自独立复核。左上用户所指橙框精确对应原标签的两条斜指，独立复核的水平候选没有被误当作用户认可。颜色、坐标及面板位置见'+link(E/'data/evidence/panel_identity_mapping.json','映射清单')+'。所有坐标是原图像素xyxy、右下端不包含。')
 lines.append('![用户裁决后处理](figures/user_adjudication4.png)\n此图保留原四面板顺序；紫实框表示可计分夹指，绿实框表示可计分前板，点框只提示可见候选、未计分，红虚框定位用户排除的水平白段。轴为原图像素、y向下。原候选图的橙/蓝表示来源，本图颜色表示处理结果，两者含义不同。没有模型输出；unknown不等于不存在。')
 lines.append('右上没有被标成“无夹指”：右侧灰色指体可见，另一根的局部灰段与把手投影相接，尚无可靠完整可见边界。现有评分约定是类别图级unknown，不能只标右指而让另一根潜在真实指体被当FP。用户指出的白段可以成为后续硬负例的语义证据，但其位置矩形右端跨过邻近指体，不能将整个矩形自动当干净负样本。')
 lines.append('左下用户确认“这些木板会动”，该证据已经记录；当前目标却是前板连把手，明文排除腔底与侧板。原图棕色表面延伸出裁切上缘，无法从此次话语确定哪条像素线分开前板和其它可移动木面。将裁切的y48或随意选出的y55当部件上界会伪造精确性，所以保留边界unknown。两项未定边界及全部可见候选见'+link(E/'data/labels/user_decision_ledger.json','逐项决策')+'。')
 lines.append('## 新旧版本及分母\n')
 lines.append(table(['计分范围','类别图数','参考框数','用途'],[['原始选中8图','11','18','原封存标签，只作历史对照'],['助手裁决v1','12','18','前一轮8图版本'],['用户语义v2','14','21','本轮新版本，仍2个unknown'],['v1与v2共同范围','12','18','两版同一对象、同一坐标、同一预测'],['v2相对v1新增','2','3','id0两根夹指、id9一块前板'],['原始与v2共同范围','10','16','另行隔离此前坐标修正及新增覆盖'],['v2相对原始新增','4','5','id9前板、id32夹指、id33/id39前板'],['原始相对v2移出','1','2','id8夹指边界仍未知']]))
 lines.append('只有四个争议类别图允许写新记录，其中两项改为known、两项补充语义但仍unknown；另外12个类别图与v1逐项相同。选集仍是4张A/B训练图、2张C已见诊断图、2张D已见诊断图，A/B开发图为0。v2中训练部分由v1的6参考框变为9框，C和D各6框不变。'+link(R/'denominator_changes.csv','逐图分母')+'与'+link(E/'configs/scoring_masks.json','固定计分范围')+'保留全部成员。')
 lines.append('## 固定保存预测的结果\n')
 lines.append('同一批224/448、种子0/1的已保存预测按固定分数≥0.5、原保存NMS IoU门槛0.3、匹配IoU≥0.5计分。原解码与此前修正解码都未重跑；修正解码指之前去掉局部峰值/数量上限的保存版本，不表示本轮改进模型。每格依次是TP/FP/FN：匹配/误检/漏检数；所有未匹配输出在known类别图上计FP。')
 rows=[]
 for var in['original','corrected']:
  for res,seed in cfg:
   trip=lambda x:f'{x["TP"]}/{x["FP"]}/{x["FN"]}'
   rows.append(['原解码'if var=='original'else'修正解码',f'{res}/s{seed}',trip(g(res,seed,var,'user_v2','common_v1_v2')),trip(g(res,seed,var,'user_v2','added_v2_vs_v1')),trip(g(res,seed,var,'user_v2','all_known_selected8'))])
 lines.append(table(['保存版本','分辨率/种子','共同18框：v1=v2','新增3框：只在v2计分','v2全部21框'],rows))
 lines.append('共同范围的结果完全不变是预期的：该范围的标签、坐标和预测都没有变化，不能当作模型稳健性的新证据。新增3框中，左上两根夹指在224修正解码与448两种解码都匹配2框；224原解码每种子仅匹配1框。右下前板在所有配置均未达到0.5匹配门槛，最佳IoU最高约0.381。新增结果来自计分覆盖改变，不是重训收益；该腕部前板在原训练标签中是unknown，原训练没有这一正例监督。')
 lines.append('![共同与新增结果分解](figures/saved_prediction_decomposition.png)\n图只展示保存的修正解码结果。横轴为分辨率/种子，纵轴为框数；灰段是固定共同范围，橙段是新纳入3框的贡献；三个面板分别为TP、FP、FN。柱顶为v2合计。它说明哪些计数来自新覆盖，不用于挑选模型；不同种子不是独立图像样本。')
 lines.append('全部版本、类别、split及计分范围见'+link(R/'all_group_metrics.csv','分组结果')+'；逐框匹配和unknown输出见'+link(R/'all_image_metrics.csv','逐图结果')+'。与原始标签的共同范围比较另见'+link(R/'ORIGINAL_COMPARISON.md','原始对照表')+'。没有对32张只读审查图重新跑开发评分或修改全40图指标。')
 lines.append('## 把手与夹指混淆是否重复\n')
 lines.append('只读查看了全部32张A/B训练与开发RGB、原封存夹指标注及完整连接上下文；没有以预测成绩筛选可疑图，也没有生成其余图的新标签。这里必须区分“整个实例认错”与“目标正确但框过宽”。外接矩形可能合理包含邻近背景；只有边界没有真实指体可见端点支持时，才判为范围问题。')
 lines.append(table(['层级','具体项','当前证据及限制'],[['用户确认的身份错误','独立复核id0水平候选、id8水平白候选','id0用户认可橙色两根；id8用户明确白段是把手。两项独立候选从未用于原模型训练，不能把独立复核错误算成训练标签错误'],['确认原框混入非指像素','训练id8左框','框内大量水平白段由用户确认为把手；不能原样保留或把独立扩大的框当夹指，另一指边界仍待定'],['确认过宽范围，非整实例身份定罪','训练id13；开发id21、id31','id13框下端跨入黑U把手；id21右界延到背景把手；id31下界超出灰指尖进入黑物体/棕背景。仅登记，未改原文件'],['此前已修正的范围问题','训练id1夹指','v1已收紧过宽框，本轮沿用，不再据分数改边界'],['疑似连接边界问题','训练id4；开发id28','近端灰指与水平白段投影相接，需确认掌部远端分界；没有确认整根夹指被换成把手'],['需要新的可见性/遮挡核验','训练id12；开发id22、id30','12/22原为unknown；30左指可见长度与visible状态存疑；不自动补框'],['本次未发现同类身份错标','其余21张A/B图','原框指向上缘或掌部远端灰指，未确认将水平白段单独作为完整夹指；不等于所有边界准确']]))
 lines.append('**开发集中没有确认“将同一类水平白段直接当成完整夹指实例”的重复案例**；但有可确认的框过宽/背景混入及待核边界，开发标签还不能作为完整审计过的金标准。训练id8是当前直接与用户把手证据关联的原框问题。上述范围错误也不能证明原标注者当时的心理原因就是部件身份混淆。完整逐图清单和原坐标见'+link(R/'AB_READONLY_AUDIT.md','只读审查账本')+'与'+link(R/'AB_readonly_findings.csv','CSV')+'。')
 lines.append('![只读标注问题例图](figures/readonly_audit_examples.png)\n四图紫框全部是原封存夹指标注，轴为原图像素。前两图呈现确认的过宽边界，后两图是疑似/待核项目；没有新预测或替代标签。它用于定位人工核验工作，不用于计算错误率。')
 lines.append('## 方法结论与最小下一步\n')
 lines.append('紧致裁剪可以隔离背景，但不能抹掉部件与父对象的连接上下文。细长、灰白的局部外观同时出现在把手和夹指上；只截取那段外观，会删除它连接掌部还是抽屉、是否与另一根手指配对的证据。必要连接本来被遮挡时，应保留unknown，而不是靠外形猜补。')
 lines.append('![紧裁与连接上下文](figures/context_vs_tight_crop.png)\n三个面板来自同一原始RGB：完整场景、紧裁白段、带掌部与邻近灰指的扩大局部。红虚框仅索引用户确认的把手位置；没有生成额外像素或假定可见连接。坐标轴为原图像素，各面板尺度不同。紧裁显示外观相似，扩大视野保留可用于身份核验的信息；它不是网络效果对比实验。')
 lines.append('这次用户证据支持建立“把手不是夹指”的硬负例和局部连接关系核验，**没有证明必须更换网络**，也没有证明增加连接模块会提升性能。最小下一步只做数据核验：先补清id1前板与id8另一根夹指的可见边界，并核对id13、id21、id31三处明显过宽框，形成带父对象上下文的5图复核材料；id4/28及12/22/30留在明确的后续队列。所有新标签另版保存、说明语义证据与边界来源；硬负例只指已确认非指的可见部分，不把混有相邻手指的整矩形直接设为负例。')
 lines.append('训练仍待另行决定。决定前应先完成将要使用的标签审计、冻结根级划分与评价口径，并使用未参与本次裁决的样本检验泛化；不应把这8图重评当成独立开发/测试，也不应同时改变标签、网络、损失和阈值后归因。本轮未执行上述下一步材料或任何训练。')
 lines.append('## 验证与限制\n')
 lines.append('新标签和只读审查账本在本轮首次读取保存预测之前冻结；裁决者有历史项目结果上下文，已披露并不声称盲审。独立数值实现复核296条可计分记录；原始标签及v1的256条旧记录完全复现；3240条分组行计数/分母核对通过；7个旧封存实验共2111文件哈希不变。其余12个类别图标签逐项未改。'+link(E/'results/verification.json','核验证据')+'可查。')
 lines.append('除这8图之外的32图仍未完成完整标注审计：本轮其中28张A/B图只做夹指身份与明显边界问题扫描，另4张C/D图未审。所有全数据旧分数继续暂定；没有外推8图修正后的成绩，没有重算开发分数，没有宣称世界模型或规划验收通过。当前仍有两项边界未知，以及旧训练/开发标签需复核；执行没有资源或权限阻塞。')
 lines.append('源码、运行记录、来源与封存见'+link(R/'RUN_COMMANDS.md','运行说明')+'、'+link(E/'configs/user_labels_and_audit_frozen.json','标签与审查冻结清单')+'、'+link(E/'configs/delivery_manifest.json','交付清单')+'。')
 (R/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
 # Detailed read-only audit; preserve exact old coordinates.
 ll=['# A/B只读身份与范围审查\n','原标签不变；无模型输出参与。此清单不是新标注，不代表全部标签审计完成。']
 for r in audit:ll.append(f'## id{r["image_id"]:02d} / {r["split"]}\n\n'+table(['项目','内容'],[['根',r['root_id']],['类别',r['status']],['原状态',r['original_finger_status']],['原框',r['original_boxes']],['证据',r['evidence']],['下一次核验',r['next_verification']]])+f'\n\n[完整原图与原框](figures/audit_id{r["image_id"]:02d}.png)')
 (R/'AB_READONLY_AUDIT.md').write_text('\n\n'.join(ll)+'\n')
 oo=['# 与原始标签的独立计分范围对照\n','每格TP/FP/FN。共同范围固定10类别图/16参考框；新增范围是v2中的4类别图/5框，移出范围为原id8夹指1类别图/2框。原标签新增范围unknown，不把其计数0解释为正确或失败。所有数字仅选中8图。']
 rows=[]
 for var in ['original','corrected']:
  for res,seed in cfg:
   trip=lambda x:f'{x["TP"]}/{x["FP"]}/{x["FN"]}'
   rows.append([var,f'{res}/s{seed}',trip(g(res,seed,var,'original','common_original_v2')),trip(g(res,seed,var,'user_v2','common_original_v2')),trip(g(res,seed,var,'user_v2','added_v2_vs_original')),trip(g(res,seed,var,'original','dropped_original_vs_v2'))])
 oo.append(table(['保存解码','配置','共同范围/原标签','共同范围/v2','新增5框/v2','移出2框/原标签'],rows));(R/'ORIGINAL_COMPARISON.md').write_text('\n\n'.join(oo)+'\n')
 (R/'RUN_COMMANDS.md').write_text('''# 运行记录

所有步骤在本机，独立项目新目录。旧模型及旧标注未改动；CPU执行，没有Torch模型调用。

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
python user_arbitration_src/prepare.py
# 逐图查看原图及原标注，不读模型输出。
python user_arbitration_src/freeze_labels.py
python user_arbitration_src/score.py
python user_arbitration_src/verify.py
python user_arbitration_src/report.py
python user_arbitration_src/seal.py
```

工作目录：`/home/robot/Documents/Codex/2026-10-13/task-4`。脚本带不覆盖断言；封存后不要原地重跑。新项目写入经过窄范围沙箱授权，未发生拒绝或绕过。source/保留实际脚本，logs/保留输出。未新采样、未读未审图片的预测计分、未重训或改阈值。
''')
 (R/'STAGE_SUMMARY.md').write_text('用户四项语义全部记录并接受；左上橙色两夹指和右下斜前板加入计分，右上另一指边界及左下前板/其它木面边界仍unknown。v1→v2为12/18→14/21（类别图/框）；共同18框的结果完全相同，新增3框单独报告。32张A/B只读扫描发现原训练id8左框混入用户确认的把手，以及训练13、开发21/31过宽范围；开发未确认同种水平白段被直接作为完整夹指的重复。疑似4/28，需新核验12/22/30，不自动改。用户语义不是像素金标准。证据支持硬负例和父对象连接核验，未证明必须换网络。零训练/前向/特征/重解码，7旧封存2111文件不变；其余32图完整审计未完成。下一步仅建议先核验5图边界，再决定训练。\n')
 shutil.copy2(__file__,E/'source'/Path(__file__).name);print('USER_ARBITRATION_REPORT_READY',str(R/'REPORT.md'))
if __name__=='__main__':main()
