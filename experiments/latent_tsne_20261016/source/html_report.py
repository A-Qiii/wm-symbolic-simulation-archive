from common import *
from plotly.offline import get_plotlyjs
from datetime import datetime, timezone

HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续隐状态 t-SNE · 已封存模型诊断</title>
<style>
:root{font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#203245;background:#f4f6f8}body{margin:0}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 12px}h2{font-size:22px;margin:30px 0 12px}p{line-height:1.75}a{color:#176b9c}.card{background:white;border:1px solid #dde4e9;border-radius:12px;padding:18px;margin:14px 0}.row{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:16px}.plot{height:650px;min-width:0}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:6px;border:1px solid #b6c4d0;border-radius:5px}label{font-size:14px}.note{color:#526778;font-size:14px}.selected{overflow-wrap:anywhere}.selected img{width:100%;height:auto;border:1px solid #d5dde4}.selected h3{margin-top:0}.selected p{font-size:14px;margin:8px 0}.badge{background:#e9f1f8;padding:4px 8px;border-radius:4px;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #dfe5eb}th{background:#eef3f7}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:16px}.gallery figure{margin:0}.gallery img{width:100%}figcaption{font-size:14px;line-height:1.6;margin:6px 0 18px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:14px}.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}.summary{background:#eaf3f8}.warning{border-left:4px solid #bb8744;padding-left:12px}@media(max-width:900px){.row,.gallery{grid-template-columns:1fr}.plot{height:520px}main{padding:12px}}
</style>
<main><h1>连续隐状态 t-SNE：现有模型的可视化诊断</h1>
<p>使用已封存的 seed 0 模型与已有真实技能边界，未重新训练、未新增仿真或测试。主图输入为交互编码器的 <b>4 个连续 token × 64 维</b>，不是查询头概率，也不是原始 DINO 特征。真实标签只用于着色、悬停与事后统计。</p>
<div class="card summary"><b>这组图支持什么？</b><p>原空间的局部邻居与持物、入屉、抽屉状态有关；但仍保留环境信息，且 AE 在持物和抽屉邻居一致率上相近。因此不能把二维分组解释为交互模型全面优于 AE、跨环境不变或规划已经成功。D 和 D2 是物理环境 D 的两次已用测试采样，不是两个物理域。</p></div>
<h2>1. 同一组状态，多种着色</h2>
<div class="card"><div class="controls"><label>表示 / 投影 <select id="model"><option value="main_xy">交互编码器 · perplexity 30</option><option value="stability_xy">交互编码器 · 稳定性检查 perplexity 15</option><option value="ae_xy">AE 重构编码器 · 独立投影</option></select></label><label>颜色 <select id="colour"></select></label><label>查看状态 <select id="subset"><option value="all">全部 338 个状态</option><option value="closed">空手 + 三块均不在屉内 + 抽屉关</option><option value="open">空手 + 三块均不在屉内 + 抽屉开</option></select></label></div>
<p class="note" id="main-note"></p><div id="main-legend" class="legend"></div><p class="note">形状：○ AB训练　■ C开发　◆ D已用测试　▲ D2已用测试。悬停显示身份和真实标签，点击点查看真实双相机。更换颜色或筛选状态保留坐标；更换投影后的轴、距离不能直接比较。</p>
<div class="row"><div id="main-plot" class="plot"></div><aside id="main-detail" class="selected"></aside></div></div>
<h2>2. 真实后继与一步预测：一次联合拟合</h2>
<div class="card"><div class="controls"><label>颜色 <select id="joint-colour"><option value="group">对应真实执行条件 / 结果</option><option value="drawer">对应真实后继抽屉状态</option></select></label><label><input id="lines" type="checkbox" checked>显示预先指定的 6 组配对线</label></div>
<p>47 次已有真实转移，每次对应 1 个真实后继、1 个对照预测、1 个修复预测，共 141 点联合拟合一次。○ 真实后继，▲ 对照预测，◆ 修复预测。预测点颜色表示应对应的真实后果，不表示预测正确。虚线连对照与真实后继，点线连修复与真实后继；线长不是原空间误差。</p><div id="joint-legend" class="legend"></div>
<div class="row"><div id="joint-plot" class="plot"></div><aside id="joint-detail" class="selected"></aside></div>
<p class="note">配对线固定选红/蓝/粉各一个 held_closed 拒绝案例和一个 held_open 正常案例，未按投影效果挑选。每个预测点只配真实执行前后 RGB；模型没有生成这些图像。未到达的 pink_2 参考链第 3 个槽位保留缺失，不补造点。</p></div>
<h2>3. 原 256 维空间的邻居检查</h2><div class="card"><p>在各模型独立的标准化 256 维表示上取 10 个邻居，排除同一 root。表内是标签一致率，不是分类准确率。频率基线为各类占比平方和；类不平衡会抬高一致率。</p><table><thead><tr><th>真实标签</th><th>交互表示</th><th>AE 表示</th><th>类频率随机基线</th></tr></thead><tbody>__METRIC_ROWS__</tbody></table><p class="note">环境一致率高于频率基线，说明存在可检测的环境信息；不能仅凭该值判断信息来源或宣称环境泄漏。语义与环境分布不完全独立，查看上面的同语义子集可帮助识别混淆，但也不是环境不变性的证明。</p></div>
<h2>4. 静态高清图</h2><div class="gallery">__GALLERY__</div>
<h2>5. 固定参数、样本与解释边界</h2><div class="card"><table><tbody>
<tr><th>抽样</th><td>450 个已有技能前/后边界候选 → 338 个真实状态、57 roots、每 root 最多 8 点。按状态键 SHA256 排序；去 21 个精确重复、80 个近重复、11 个超额点。未按标签或投影挑样本。</td></tr>
<tr><th>分布</th><td>A 51 / B 51 / C 54 / D 182；其中 D 93 点来自 D 批次、89 点来自 D2 批次。AB训练102、C开发54、D已用测试93、D2已用测试89。</td></tr>
<tr><th>标准化与 PCA</th><td>均值/标准差仅用入选 AB 训练点拟合；PCA 50 维用入选无标签点拟合。联合图复用交互表示的 AB 标准化，再对 141 点联合做 PCA。使用测试点进行可视化坐标拟合仅是事后传导式诊断，未用于模型选择。</td></tr>
<tr><th>t-SNE</th><td>模型 seed 0；投影 seed 42；perplexity 30；PCA 初始化；学习率 auto（实际 50）；early exaggeration 12；最多 1500 次迭代；Barnes–Hut angle 0.5，欧氏距离。稳定性图仅将 perplexity 改为预定的 15。</td></tr>
<tr><th>稳定性</th><td>两次交互投影平均 10 邻居重合 81.9%；这描述局部布局稳定程度，不代表预测或规划有效。</td></tr>
<tr><th>环境</th><td>A/B/C/D 是 CALVIN 的不同实际场景域；D2 仍用场景 D。空手、三块均不在屉内、抽屉关闭子集 47 点（A9/B7/C8/D23）；打开子集 61 点（A9/B9/C8/D35）。子集复用主坐标，不重新拟合。</td></tr>
</tbody></table><p class="warning">t-SNE 的轴无物理单位；全局间距、簇大小和密度不宜作定量解释。两个点很近不代表符号完全正确，两个簇很远也不代表物理上差异更大。本报告是对已用数据的探索性诊断，不能替代独立预测、闭环与泛化评估。</p><p>完整采样记录、原空间统计、模型哈希和投影参数位于相邻实验目录 <code>experiments/latent_tsne_20261016/</code>。本地说明：<a href="README.md">README.md</a>；运行与审计配置另见实验目录。页面及全部图像均可离线查看。</p></div>
</main><script src="assets/plotly.min.js"></script><script>
const data=__DATA__;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const layout={margin:{l:65,r:15,t:25,b:55},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui,Noto Sans CJK SC,sans-serif',color:'#26394a'},xaxis:{title:{text:'t-SNE 1（无物理单位）'},zeroline:false,gridcolor:'#eef1f4'},yaxis:{title:{text:'t-SNE 2（无物理单位）'},zeroline:false,gridcolor:'#eef1f4'},showlegend:false,hovermode:'closest'};
const cfg={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:3}};
function legend(id,values,pal){$(id).innerHTML=Object.entries(pal).filter(([v])=>values.includes(v)).map(([v,c])=>`<span><i class="swatch" style="background:${c}"></i>${esc(v)} (${values.filter(x=>x===v).length})</span>`).join('');}
function showMain(n){$('main-detail').innerHTML=`<h3>${esc(n.id)}</h3><span class="badge">环境 ${esc(n.environment)} · ${esc(n.sampling_group)}</span><p><b>真实标签：</b>${esc(n.truth)}</p><img src="${esc(n.image)}" alt="${esc(n.id)} 固定与腕部真实相机"><p><b>root：</b>${esc(n.root)}<br><b>状态键：</b>${esc(n.key)}<br><b>采样批次：</b>${esc(n.batch)}</p><p><b>技能边界来源：</b>${n.roles.map(r=>esc(r.pair_id)+' · '+esc(r.role)+' · '+esc(r.skill)).join('<br>')}</p>`;}
function maskMain(n){const q=$('subset').value;if(q==='all')return true;return n.labels.holding==='空手'&&['red_in','blue_in','pink_in'].every(k=>n.labels[k]==='不在屉内')&&n.labels.drawer===(q==='closed'?'关闭':'打开');}
async function drawMain(){const field=$('colour').value,xy=data[$('model').value],ids=data.main_nodes.map((_,i)=>i).filter(i=>maskMain(data.main_nodes[i]));const n=ids.map(i=>data.main_nodes[i]);const tr={type:'scatter',mode:'markers',x:ids.map(i=>xy[i][0]),y:ids.map(i=>xy[i][1]),customdata:n,text:n.map(p=>`${p.id} · ${p.sampling_group}<br>环境 ${p.environment} · root ${p.root}<br>${p.truth}`),hovertemplate:'%{text}<extra></extra>',marker:{color:n.map(p=>data.palette[field][p.labels[field]]),symbol:n.map(p=>p.symbol),size:9,opacity:.9,line:{color:'white',width:.5}}};const xr=[Math.min(...xy.map(p=>p[0])),Math.max(...xy.map(p=>p[0]))],yr=[Math.min(...xy.map(p=>p[1])),Math.max(...xy.map(p=>p[1]))];const pad=r=>[r[0]-(r[1]-r[0])*.08,r[1]+(r[1]-r[0])*.08];await Plotly.react('main-plot',[tr],{...layout,xaxis:{...layout.xaxis,range:pad(xr)},yaxis:{...layout.yaxis,range:pad(yr)}},cfg);legend('main-legend',n.map(p=>p.labels[field]),data.palette[field]);$('main-note').textContent=`当前显示 ${ids.length} 个真实技能边界。颜色：${data.titles[field]}。`;$('main-plot').removeAllListeners('plotly_click');$('main-plot').on('plotly_click',e=>showMain(e.points[0].customdata));if(n.length)showMain(n[0]);}
function showJoint(n){$('joint-detail').innerHTML=`<h3>${esc(n.id)} · ${esc(n.kind_name)}</h3><span class="badge">${esc(n.group)}</span><p><b>对应真实后继：</b>${esc(n.truth)}</p><p><b>该点经冻结查询头：</b>${esc(n.query)}</p><p>真实执行前</p><img src="${esc(n.before_image)}" alt="真实执行前双相机"><p>真实执行后</p><img src="${esc(n.image)}" alt="真实执行后双相机"><p>${esc(n.image_note)}</p><p><b>root：</b>${esc(n.root)}<br><b>pair：</b>${esc(n.pair_id)}<br><b>技能：</b>${esc(n.skill)}<br><b>条件 / 实际结果：</b>${esc(n.condition)} / ${esc(n.status)}</p>`;}
async function drawJoint(){const f=$('joint-colour').value,pal=f==='group'?data.group_palette:data.palette.drawer,xy=data.joint_xy,n=data.joint_nodes,tr=[];if($('lines').checked)for(const l of data.lines){tr.push({type:'scatter',mode:'lines',x:[xy[l.a][0],xy[l.b][0]],y:[xy[l.a][1],xy[l.b][1]],line:{color:'#a2adb7',width:1.2,dash:l.kind==='canonical_control'?'dash':'dot'},hoverinfo:'skip'});}tr.push({type:'scatter',mode:'markers',x:xy.map(p=>p[0]),y:xy.map(p=>p[1]),customdata:n,text:n.map(p=>`${p.id} · ${p.kind_name}<br>${p.group}<br>root ${p.root}<br>对应真实后继：${p.truth}`),hovertemplate:'%{text}<extra></extra>',marker:{color:n.map(p=>pal[p[f]]),symbol:n.map(p=>p.symbol),size:10,opacity:.9,line:{color:'white',width:.5}}});await Plotly.react('joint-plot',tr,layout,cfg);legend('joint-legend',n.map(p=>p[f]),pal);$('joint-plot').removeAllListeners('plotly_click');$('joint-plot').on('plotly_click',e=>{if(e.points[0].customdata)showJoint(e.points[0].customdata);});showJoint(n[0]);}
for(const [key,title]of Object.entries(data.titles)){$('colour').add(new Option(title,key));}
for(const id of ['model','colour','subset'])$(id).addEventListener('change',drawMain);
for(const id of ['joint-colour','lines'])$(id).addEventListener('change',drawJoint);
Promise.all([drawMain(),drawJoint()]).then(()=>document.documentElement.dataset.ready='true').catch(e=>{document.documentElement.dataset.error=String(e);throw e;});
</script></html>'''

def main():
 payload=read(E/'results/interactive_payload.json')
 metrics=read(E/'results/original_space_neighbours.json')
 rows=[]
 for field in ['holding','red_in','blue_in','pink_in','drawer','environment']:
  a=next(x for x in metrics if x['field']==field and x['model']=='interaction');b=next(x for x in metrics if x['field']==field and x['model']=='ae')
  rows.append(f"<tr><td>{payload['titles'][field]}</td><td>{a['agreement']:.1%}</td><td>{b['agreement']:.1%}</td><td>{a['class_frequency_random_baseline']:.1%}</td></tr>")
 captions={
 'main_states.png':'同一 338 点与同一投影，依次按真实持物、红/蓝/粉入屉、抽屉状态和物理环境着色；形状区分训练、开发与已用测试批次。',
 'successor_joint.png':'47 组真实后继和两臂预测联合拟合；形状区分真实/对照/修复，颜色来自对应真实结果。细线仅连固定 6 组案例，不能把二维线长当误差。',
 'environment_and_batch.png':'同一坐标左按物理环境着色，右按数据用途与批次着色；D 与 D2 都是物理环境 D。',
 'same_state_cross_domain.png':'空手、三块均未入屉且抽屉关闭/打开两个固定语义子集，按物理环境着色。浅灰为其他全部状态，坐标未重新拟合。',
 'ae_comparison.png':'AE 重构表示对照，同 338 点但另做独立投影；颜色含义与主图相同，不可跨模型比较轴或簇间距离。',
 'stability.png':'交互表示固定 seed 42，左 perplexity 30、右 15；按真实抽屉状态着色。两张都保留，未按视觉效果选择。'}
 gallery=''.join(f'<figure><a href="{f}"><img loading="lazy" src="{f}" alt="{c}"></a><figcaption><a href="{f}">{f}</a>：{c}</figcaption></figure>' for f,c in captions.items())
 (R/'assets/plotly.min.js').write_text(get_plotlyjs())
 output=HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')).replace('__METRIC_ROWS__',''.join(rows)).replace('__GALLERY__',gallery)
 (R/'index.html').write_text(output)
 print('HTML_READY',R/'index.html',len(output),flush=True)

if __name__=='__main__':main()
