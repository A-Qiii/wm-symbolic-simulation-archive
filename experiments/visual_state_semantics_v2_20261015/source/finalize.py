from common import *
import sys,subprocess,collections,shutil,html
R=P/'reports/state_semantics_v2_20261015';EC=P/'experiments/visual_state_semantics_v2_hardcontrast_20261015'

def main():
 # Actual separate CPU CLI processes. Evaluator labels only decorate report after output.
 commands=read(R/'tables/query_commands.json');outputs=[]
 for error in read(R/'tables/illustrated_In_errors.json'):
  commands.append({'window_id':error['window_id'],'query':error['fact'].replace('In(', 'In(obj_'),'truth_for_report_only':False})
 for version,ep in [('initial',E),('single_repair',EC)]:
  for r in commands:
   cmd=[sys.executable,str(ep/'source/query_demo.py'),'--window-id',r['window_id'],'--query',r['query']];run=subprocess.run(cmd,text=True,capture_output=True,check=True);out=json.loads(run.stdout);assert out['GT_read_at_inference'] is False;outputs.append({'version':version,'command':cmd,'output':out,'report_comparison_only':r})
 save(R/'tables/query_cli_outputs.json',outputs)
 from learn_actions import search
 ops=read(E/'configs/learned_rules9.json')['operators'];tests=[]
 for st in [[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]:
  result=search(st,[(3,1),(6,1)],ops);assert result['plan'] is None;tests.append({'input9':st,'goal':'In(red,drawer) AND Closed(drawer)','result':result})
 mid=[x for x in outputs if x['report_comparison_only'].get('label_class_for_report_only')=='Intermediate']
 assert all(x['output']['decoded_current_state']['Closed(drawer)']==0 and x['output']['decoded_current_state']['Open(drawer)']==0 and x['output']['decoded_current_state']['Intermediate(drawer)']==1 for x in mid)
 save(R/'tables/planner_query_checks.json',{'empty_closed_open_mid_do_not_invent_place_plan':tests,'actual_intermediate_query_does_not_negate_Open_into_Closed':True,'versions_checked':2,'actual_CLI_calls':len(outputs),'passed':True})
 # Duplicate inputs are audited without pretending windows are independent.
 inp=dict(np.load(E/'data/inputs.npz'));meta=read(E/'data/metadata.json');groups=collections.defaultdict(list)
 for i,r in enumerate(meta):
  h=hashlib.sha256(b''.join(inp[k][i].tobytes()for k in sorted(inp))).hexdigest();groups[h].append(i)
 duplicates=[{'input_sha256':h,'windows':[{k:meta[i][k]for k in ['window_id','root_id','split','pool']}for i in ids]}for h,ids in groups.items()if len(ids)>1];cross=[r for r in duplicates if len({x['split']for x in r['windows']})>1];save(R/'tables/input_duplicate_audit.json',{'windows':len(meta),'unique_input_tensors':len(groups),'duplicate_groups':duplicates,'cross_split_exact_duplicate_groups':cross,'same_root_windows_not_independent':True})
 d=read(E/'results/development_evaluation.json');checks=d['gate_checks'];summary=read(R/'tables/report_summary.json');qexample=commands[2];readme=f'''# 本机状态语义第二版

[结论与全部图表](REPORT.md) · [真实视频与图表](gallery.html)

本轮使用AB/C进行开发，未启动新D。是否通过开发门槛：{d['new_independent_acceptance_permitted_by_frozen_gate']}。不能把171个采集回合称为模型闭环验收：动作由固定脚本／执行器执行，模型在公开保存窗口上训练和推理。

## 实际执行一次三状态查询

```bash
source /home/robot/research/projects/wm_symbolic/world_model_env.sh
python {E}/source/query_demo.py \\
  --window-id {qexample['window_id']} \\
  --query 'Intermediate(drawer)' --goal-object red
```

可替换查询为Closed(drawer)、Open(drawer)、Holding(obj_red)、In(obj_red,drawer)。输入只从公开视觉／本体特征读取，不加载监督文件。输出包含独立Closed目标和只允许受支持学得规则的搜索。无规则支持时返回无计划，不用GT或手写效果补齐。实际命令输出见tables/query_cli_outputs.json。

## 两版结果与唯一次修正

初版与修正版都完成1200步，共2400步；C完整九事实都是165/209。修正版没有通过门槛，也未晋升部署。它只加AB同根正负In配对损失，其他训练条件固定，基础抽样序列逐项相同。修正版源码、模型和日志位于`{EC}`，实际同窗CLI对照保存在tables/query_cli_outputs.json。把上面命令的实验目录改为该目录即可查询修正版。

## 已执行的脚本和顺序

- `prepare.py`冻结根、采集和标签协议；`collect.py`保存最早A首批，`collect_v2_1.py`保留首批并继续全量真实采集；`audit.py`从完整物理trace独立重算标签、检查公开输入、物理步数和边界测试。
- `cache_inputs.py`：630新窗口重编码，旧312仅复用Holding监督；现有DINO缓存复用，无下载。首次锁重复申请失败日志cache.log保留，cache_lock_fixed.log为修正后的实际运行；修正仅移除外层重复锁，RGBEncoder原锁仍在。
- 初版`train.py`和独立修正版`train.py`：各1200更新、固定检查点／选择规则，索引仅AB；各自训练前冻结，C只作开发，不称盲测。
- `learn_actions.py`：AB真实成功／失败学习九事实算子，不支持的簇不能供搜索；`analyze_development.py`：C评估、同根摆放与接触诊断、完全相同九事实前态的结果冲突审计。
- `report.py`、`media_and_queries.py`、`finish_report.py`、`finalize.py`：CPU图表、真实视频和交付核查。

物理采集／视频用`env.sh`，DINO／训练／CLI用`world_model_env.sh`，同一GPU串行。现有输出有防覆盖断言，不要直接重跑训练／采集目录。新试验须独立目录和冻结协议。任何模型、标签、门槛变化需作为新版本保留本轮证据。

## 目录

|路径|内容|
|---|---|
|`{E}/configs`|初始和v2.1标签协议、哈希、模型设计、选定模型、学得算子|
|`{E}/source`|可复用源码，原版本保留|
|`{E}/data`|公开模型特征、监督目标、元数据、重算标签；GT不进入输入|
|`{B}`|171回合原始RGB和_private完整接触／姿态trace，含所有失败|
|`{E}/models`|4个检查点，均保留|
|`{E}/logs`|采集、首次锁失败、缓存、训练、评估、报告日志|
|`{EC}`|唯一修正协议、AB配对、4个检查点及完整训练日志|
|`tables`|开发门槛、接触与摆放对照、状态冲突、查询输出、输入重复和交付清单|
|`videos`|同根堆叠受阻、并排成功、C固定4技能链，真实相机3倍播放|

上一轮七事实“6/6”不等于物理关严；正确严格结果5/6已保留并解释。此轮使用独立Closed表达，不继续沿用¬Open的完成判断。
''';(R/'README.md').write_text(readme)
 text=(R/'REPORT.md').read_text();text=text.split('\n## 实际查询和文件核查')[0];text+='\n## 实际查询和文件核查\n\n已对两版分别实际执行Closed、Open、Intermediate、正例In及两类错误In，共12条CLI查询，原始命令及输出在tables/query_cli_outputs.json。共享查询的实体别名与对象置换检查在训练末尾执行通过；Closed/Open/Intermediate概率归一化且三者互斥，未由Open取反构造Closed。\n\n输入精确重复审计：'+str(len(meta))+'窗口对应'+str(len(groups))+'个不同的标准化输入张量，跨训练／开发精确重复组'+str(len(cross))+'；重复和同根窗口不当独立样本。\n\n首次特征缓存因外层与既有RGBEncoder重复获取同一GPU锁而在编码前失败；移除外层重复锁后重启，未产生额外训练更新、未修改模型设计或语义。失败日志和修正哈希保留。\n';(R/'REPORT.md').write_text(text)
 vids=read(R/'videos/manifest.json');cards=''.join(f'<article><h2>{html.escape(v["title"])}</h2><video controls preload="metadata" src="videos/{Path(v["path"]).name}"></video><p>真实静态相机与腕部相机，3倍播放，末帧展示独立GT评分。不是模型实时闭环。</p></article>'for v in vids);figs=''.join(f'<article><a href="figures/{p.name}"><img src="figures/{p.name}"></a></article>'for p in sorted((R/'figures').glob('*.png')))
 (R/'gallery.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态语义第二版：真实开发证据</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#f3f5f7;color:#193b48;font:17px/1.6 sans-serif}article{padding:20px;background:white;border-radius:10px;margin:24px 0}img,video{width:100%;height:auto}h1{font-size:30px}h2{font-size:22px}a{color:#176d83}</style><h1>状态语义第二版：真实开发证据</h1><p>Closed、Open、中间状态分开；In要求完整几何容纳与承重支撑。27根、171回合、108真实调用、两版各1200训练步；仅AB/C开发，新D为0。C完整状态两版均165/209，未通过门槛。</p><p><a href="REPORT.md">详细结论</a> · <a href="README.md">执行命令</a> · <a href="tables/development_gate.json">冻结门槛结果</a></p>'+cards+figs+'</html>')
 # Old sealed report files and original freeze references must remain unchanged.
 oldreport=P/'reports/visual_predicator_minimal_20261015';olddelivery=read(oldreport/'delivery_manifest.json');assert all(sha(oldreport/r['path'])==r['sha256']for r in olddelivery['files']);oldref=read(P/'reports/state_readout_review_20261015/source_hashes.json');assert all(sha(p)==h for p,h in oldref.items());checks={}
 for name in ['collection_frozen.json','collection_v2_1_frozen.json']:
  for p,h in read(E/'configs'/name)['sha256'].items():checks[p]=sha(p)==h
 for ep in [E,EC]:
  for p,h in read(ep/'configs/training_frozen.json')['frozen_sha256'].items():checks[p]=sha(p)==h
  cp=read(ep/'configs/reader_frozen.json');checks[cp['path']]=sha(cp['path'])==cp['sha256']
 assert all(checks.values());save(R/'tables/integrity.json',{'passed':True,'old_report_files_unchanged':len(olddelivery['files']),'older_source_files_unchanged':len(oldref),'this_version_frozen_assets':len(checks),'files':checks})
 source=[]
 for base in [E,EC,B]:
  for p in sorted(base.rglob('*')):
   if p.is_file()and'__pycache__'not in p.parts:source.append({'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(R/'tables/source_data_manifest.json',{'project':str(P),'files':source,'bytes':sum(r['bytes']for r in source),'free_bytes':shutil.disk_usage(P).free})
 files=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(R.rglob('*'))if p.is_file()and p.name!='delivery_manifest.json'and'__pycache__'not in p.parts];save(R/'delivery_manifest.json',{'created_utc':now(),'files':files,'source_files':len(source),'new_D':0,'training_updates':2400,'training_runs':2,'root_groups':27,'actual_physics_calls':108,'integrity_checks':checks,'development_gate':d['gate_checks'],'development_gate_after_repair':read(EC/'results/development_evaluation.json')['gate_checks']});print(json.dumps({'report':str(R/'REPORT.md'),'source_files':len(source),'source_bytes':sum(r['bytes']for r in source),'old_assets_unchanged':True,'D_runs':0},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
