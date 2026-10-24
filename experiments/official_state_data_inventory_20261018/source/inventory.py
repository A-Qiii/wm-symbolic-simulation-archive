"""Read-only source inventory; writes a separate report, no download or simulation."""
from pathlib import Path
import os,json,hashlib,datetime,subprocess,zipfile,shutil
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_state_data_inventory_20261018';R=P/'reports/official_state_data_inventory_20261018'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
def main():
 assert not (E/'results/inventory.json').exists()
 for p in [E/'source',E/'results',E/'configs',R]:p.mkdir(parents=True,exist_ok=True)
 archive=P/'cache/calvin_debug_dataset.zip';raw=P/'data/raw/calvin_debug_dataset';assets=json.loads((P/'source/assets.json').read_text());original=next(a for a in assets if a['archive']==str(archive));digest=sha(archive);assert digest==original['sha256']
 records=[];refs={str(archive):digest,str(P/'source/assets.json'):sha(P/'source/assets.json')}
 for split in ['training','validation']:
  d=raw/split;files=sorted(d.glob('episode_*.npz'));scene=np.load(d/'scene_info.npy',allow_pickle=True).item();eps=np.load(d/'ep_start_end_ids.npy');ids=[int(x.stem.split('_')[1])for x in files];assert ids==list(range(int(eps[0,0]),int(eps[0,1])+1))
  first=np.load(files[0],allow_pickle=False)
  rec={'split':split,'path':str(d),'scene_info':scene,'episode_ranges_inclusive':eps.tolist(),'frame_npz_count':len(files),'frame_ids_range':[ids[0],ids[-1]],'one_independent_episode':len(eps)==1,'all_file_bytes':sum(q.stat().st_size for q in d.rglob('*')if q.is_file()),'npz_bytes':sum(q.stat().st_size for q in files),'merged_config_scene':'calvin_scene_D','merged_config_table':'calvin_table_D/urdf/calvin_table_D.urdf','legacy_config_and_overrides_scene':'calvin_scene_A','eligible_AB_training':False,'reason':'scene_info and merged_config both identify D; legacy config/overrides conflicting and not sufficient provenance','first_frame':str(files[0]),'first_frame_schema':{k:{'shape':list(first[k].shape),'dtype':str(first[k].dtype)}for k in first.files},'original_contact_forces_or_full_physics_state_available':False}
  for q in [d/'scene_info.npy',d/'ep_start_end_ids.npy',d/'.hydra/merged_config.yaml',d/'.hydra/config.yaml',d/'.hydra/overrides.yaml',files[0]]:refs[str(q)]=sha(q)
  records.append(rec)
 with zipfile.ZipFile(archive)as z:ziprec={'members':len(z.infolist()),'compressed_member_bytes':sum(x.compress_size for x in z.infolist()),'uncompressed_member_bytes':sum(x.file_size for x in z.infolist()),'zip64_allowed':True,'crc_all_members_not_retested_this_round':'bootstrap source recorded complete testzip and extraction; this round SHA matches that saved archive'}
 candidates=[];scopes=[P/'cache',Path('/home/robot/research/data'),Path('/home/robot/Downloads'),Path('/home/robot/research/projects/robot_state')]
 for root in scopes:
  if not root.exists():continue
  for here,dirs,files in os.walk(root):
   dirs[:]=[d for d in dirs if d not in ['.venv','.git','node_modules','__pycache__','.envs','models']]
   for name in files:
    if(name.endswith(('.zip','.tar','.tgz','.tar.gz','.part'))and any(k in name.lower()for k in ['calvin','task_abc','debug_dataset']))or name in ['ep_start_end_ids.npy','scene_info.npy']:
     q=Path(here)/name;candidates.append({'path':str(q),'bytes':q.stat().st_size})
 commits={n:subprocess.check_output(['git','-C',str(P/n),'rev-parse','HEAD'],text=True).strip()for n in ['vendor/calvin','vendor/calvin/calvin_env','vendor/hulc']}
 space={str(d):dict(zip(['total_bytes','used_bytes','free_bytes'],shutil.disk_usage(d)))for d in [P,Path('/tmp')]}
 inventory={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_official_AB_found':False,'search_scopes':[str(x)for x in scopes]+[str(P/'data/raw')],'search_exclusions':['environments','git','node_modules','model_weights'],'not_claimed_entire_machine_search':True,'archive':original|{'current_sha256_verified':digest,'current_bytes':archive.stat().st_size},'zip_directory':ziprec,'splits':records,'candidate_files':candidates,'vendor_commits':commits,'storage':space,'new_download_bytes':0,'simulation_calls':0,'model_updates':0,'source_references_sha256':refs,'external_findings_received_not_retested_locally':{'ABC_zip_bytes':555309812705,'range_suffix_65558_status':206,'zip64_central_directory_bytes':229211511,'source':'parent research handoff, retained as reported evidence only'}}
 write(E/'results/inventory.json',inventory)
 proposal={'status':'bounded_retrieval_authorized_by_parent_20261018','purpose':'state readout coverage and label quality only; not controller transition training','sampling_seed':170818,'proposed_scene_episode_count':{'A':2,'B':2},'episode_selection':'Select independent official episode ranges using scene_info and ep_start_end_ids before inspecting state labels; no C/D; preserve root isolation','frames':'8 evenly spaced anchors per episode plus next frame, at most64 frames; never use 64-frame language window as skill boundary','approved_hard_download_budget_bytes':512*1024**2,'central_directory_budget_bytes':229211511,'budget_source':'parent explicit approval, 2026-10-18', 'no_model_training':True,'before_download':'Use parent-verified official ZIP64 offsets and ETag. Cache bounded229211511-byte central directory within total512MiB. Verify scene mapping, official episode boundaries and config first. On HTTP200 or mismatched Content-Range stop before reading body. Validate ZIP member name, CRC, compression ratio/size bounds and hashes. If scope exceeds cap, stop.','color_mapping':'Check per-scene config order, URDF dimensions and RGB projections; do not blindly apply issue41 swap. Any ambiguity makes object labels unknown.','restoration':'Compare original RGB/depth and scene/robot slots against restored renderer/geometry; record immediately after reset and one native physics tick; quantify per-object position/orientation, joints, RGB mismatch and contacts/forces. A reset is not a full physical snapshot.','labels':'Keep same strict Holding/In definitions. No gripper-closed proxy; absent/unstable restored contacts do not prove original negative. Unknown mask for unsupported or unstable evidence; geometric In only where mapping, containment/support evidence and restore checks jointly suffice. No speculative thresholds frozen here.','split':'Official episode is isolation key; selected pilot roots stay quarantine/evidence-only until data-label review. C/D remain excluded from training.','training_updates':0,'allowed_now':'bounded official member retrieval and label-quality pilot authorized; none performed during this inventory'}
 write(E/'configs/proposed_pilot.json',proposal)
 avg=sum(x['npz_bytes']for x in records)/sum(x['frame_npz_count']for x in records)
 text='# 官方CALVIN状态数据：本地盘点与最小小试提案\n\n已完成上一轮Holding诊断与端点附录封存。本机已查范围没有发现官方A/B轨迹；现有官方轨迹只有debug数据，两个split均由scene_info和merged_config指向D，不能用于本次AB状态训练。此次仅只读盘点，下载0字节、物理恢复0次、模型更新0次。\n\n'
 text+='| 本地资源 | 实测内容 | 本轮用途 |\n| --- | --- | --- |\n'
 text+=f"| debug ZIP | {archive.stat().st_size:,}字节；SHA256 `{digest}` | 已有缓存，禁止重复下载 |\n"
 for rec in records:text+=f"| debug {rec['split']} | D；{rec['frame_npz_count']}帧，官方episode区间{rec['episode_ranges_inclusive']}，仅1个独立episode | 格式检查，不纳入AB训练 |\n"
 text+='| 官方AB/ABC原始play | 已查目录未发现 | 已获512MiB总量小取许可，盘点时尚未下载 |\n| HULC_ABC_D.zip | 执行器权重，不是ABC原始轨迹 | 不计为AB数据 |\n| 既有自采ABC/D | 项目历史固定执行器数据 | 保持原封存，不伪装成官方play |\n\n'
 text+='查找范围为项目data/raw、cache、research/data、旧robot_state项目和Downloads；不声称扫描整台机器。旧robot_state仅只读文件名检查，未改动。\n\n'
 text+='## 场景证据及不能相信的捷径\n\n两个split的`scene_info.npy`明确只有`calvin_scene_D`，`merged_config.yaml`场景名和桌体URDF也都是D。旧`config.yaml`与`overrides.yaml`仍写A，说明仅凭残留Hydra字段或文件夹名会误分场景。当前以实际数据元信息与合并配置的交叉证据排除AB用途；未假装已经做了原始RGB与恢复渲染验证。\n\n'
 text+='原始帧有RGB、深度、robot_obs、scene_obs、动作，没有保存接触力、物体速度或完整物理引擎快照。首帧中动作/状态为float64，不能盲用文档float32假设。`episode_XXXXXXX.npz`是一帧，独立episode应使用`ep_start_end_ids.npy`区间，而不是将每个npz算独立轨迹。\n\n'
 text+='[官方数据说明](https://github.com/mees/calvin/blob/main/dataset/README.md)提供ABC→D整包入口及数据结构；文档标注517GB，不代表必须整包下载。[issue #41](https://github.com/mees/calvin/issues/41)有用户报告A的红/粉与C的红/蓝槽位互换，当前只能作为待核验线索，不能据此全局交换颜色。实际小试须结合每场景配置顺序、物体尺寸和RGB投影核验。\n\n'
 text+='## 下载规模与空间\n\n'
 for d,v in space.items():text+=f"- `{d}`：可用{v['free_bytes']:,}字节（{v['free_bytes']/2**30:.1f}GiB），总{v['total_bytes']/2**30:.1f}GiB。\n"
 text+='\n父线程调研已报告ABC ZIP为555,309,812,705字节、极小Range返回206；ZIP64中央目录229,211,511字节（约218.6MiB）。这些是转交的外网测量，本机未重复请求。整包超过当前/home剩余空间，且不在本轮授权规模；后续中央目录仅按已批准的固定219MiB范围读取，计入512MiB总上限。\n\n'
 text+=f"按现有debug单帧NPZ平均{avg/1024:.1f}KiB估计，64帧约{64*avg/2**20:.1f}MiB，仅供预算参考，不当作ABC实测体积。父线程现已批准总量不超过512MiB（含中央目录）的小取和标签质量验证；复用其偏移/ETag信息。先取场景映射、官方episode边界和配置，再按实际压缩大小选少量AB片段。盘点时尚未下载；任何200响应、Content-Range不符或越限都在读取正文前停止。未经验证的对象pickle只反汇编，不执行。\n\n"
 text+='## 最小可行小试（已批准有限获取，不训练）\n\n| 步骤 | 固定内容 | 为什么需要 |\n| --- | --- | --- |\n| 场景/episode筛选 | 官方A、B各2个独立episode，seed170818；先按元数据筛，排除C/D | 保留episode隔离，不能把ABC包全部当训练 |\n| 小帧样本 | 每episode均匀8个锚点及后继1帧，最多64帧；不按成功标签挑选 | 足够先检查格式、恢复和标签不确定性，不能证明覆盖齐全 |\n| 颜色→槽位 | 场景配置、URDF大小、原RGB与几何投影共同核验 | 防止配置顺序或版本差异错标对象 |\n| 恢复审计 | 同一帧重置后立即测量，再单独一步原生物理；保存位姿/关节漂移、RGB差异、接触对及力变化 | 恢复不是历史完整物理快照，contact重建可能产生伪证据 |\n| 严格标签/掩码 | 保持Holding/In定义；缺接触或恢复不稳标未知，不把夹爪闭合当抓稳 | 避免把play粗动作标签包装成高质量状态真值 |\n| 数据去向 | 所有小试根先隔离为标签审查数据；不训练、不生成固定技能转移 | 本轮只验证状态识别数据是否可接入 |\n\n'
 text+='本机无可用AB；下一步复用已批准的官方ZIP64偏移小取元数据，再验证AB片段标签，当前没有算力或权限阻塞。没有改网络、驱动、环境或模型，也没有从debug的D帧凑AB训练样本。\n\n'
 text+=f"[原始盘点]({E}/results/inventory.json)含来源URL、缓存SHA、场景元数据、源代码版本和空间字节数；[小试提案]({E}/configs/proposed_pilot.json)保存抽样种子、预算及未知掩码原则。[端点诊断附录](../holding_endpoint_exploration_20261018/APPENDIX.md)已独立封存44文件；[完整Holding报告](../holding_readout_diagnostic_20261018/REPORT.md)封存64文件，两者结论未改。\n"
 (R/'INVENTORY.md').write_text(text);shutil.copy2(__file__,E/'source/inventory.py')
 files=[x for root in [E,R]for x in root.rglob('*')if x.is_file()]
 write(E/'configs/delivery_manifest.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'sha256':{str(x):sha(x)for x in files}})
 print('OFFICIAL_INVENTORY_DONE',len(files),'files','AB found:False','debug D frames:',sum(x['frame_npz_count']for x in records),'no downloads')
if __name__=='__main__':main()
