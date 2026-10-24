from common import *
import fcntl,subprocess
from PIL import Image
def main():
 verify();guard('seal');assert read(E/'results/verification.json')['passed'];assert read(E/'results/pipeline_status.json')['status']=='complete'
 attempts=read(E/'results/training_attempts.json');assert len(attempts)==4 and all(x['status']=='complete'and x['start_update']==0 for x in attempts)
 gpu=(P/'runs/gpu.lock').open('r+');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_UN);gpu.close()
 roots=[]
 for seed in [0,1]:
  for arm in ARMS:
   a=dict(np.load(E/f'data/predictions_{arm}_seed{seed}_dev.npz'));m=(a['mode']=='current')&(a['step']==0);v=a['prob'][m];cat=a['cat'][m];pred=np.stack([v[:,s].argmax(-1)for s in [slice(0,8),slice(8,16),slice(16,24),slice(24,27),slice(27,30)]],1)
   roots.append({'seed':seed,'arm':arm,'independent_roots':20,'root_boundary_records':60,'all5_correct':int((pred==cat).all(1).sum()),'object_field_accuracy':float((pred[:,:3]==cat[:,:3]).mean())})
 save(E/'results/initial_state_diagnostic.json',{'utc':now(),'input':'already saved current predictions atstep0','new_model_inference':0,'new_fits':0,'records':roots,'interpretation':'Many errors already present at root observation. Not all later errors are accumulated rollout errors; no unique module attribution or planning success inference.'})
 pdf=R/'RSSM编码器梯度对照_20261019.pdf';info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);assert 'Pages:           5'in info
 text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True);assert all(t in text for t in ['0.069','0.910','22/300','11/300','20/40/60'])
 images={}
 for p in(R/'figures').glob('*.png'):
  with Image.open(p)as im:im.verify()
  with Image.open(p)as im:images[p.name]=list(im.size)
 checks={'utc':now(),'PDF_pages':5,'PDF_text_extracts_numbers_and_Chinese':True,'PDF_font_parser_warning_resolved':True,'PNG_dimensions':images,'visual_review':'All3 scientific figures and PDF first/last pages inspected; final PDF first and second pages re-rendered after output-only corrections. No label clipping observed.','training_process_exit':0,'pipeline_process_exit':0,'gpu_lock_released':True,'training_attempts':4,'restarted_or_duplicate_main_fits':0,'report_only_changes':'Added full-state tradeoffs and gap interpretation from saved outputs, numeric text wrapping, font embedding. No evaluation/fit changes.'}
 save(E/'results/delivery_checks.json',checks)
 handoff=f'''# 完成本轮RSSM直接未来梯度对照

四条主训练各6000更新、四个固定线性探针各600更新已完成；主训练没有重启或重复。FP32/BF16梯度预检、2213项冻结哈希、预测分母、配对初始/采样以及主要统计独立复算通过。训练和后处理进程均退出0，GPU锁已释放。

主要结果：线性查询四字段NLL差（未来梯度组减控制）seed0 -0.108690，95%CI[-0.176,-0.042]；seed1 -0.187877，CI[-0.296,-0.088]。三步自由预测差seed0 +0.068794，CI[-1.007,1.018]；seed1 -0.909802，CI[-1.555,-0.324]。平均标签查询收益没有稳定转成完整状态正确：probe五字段全对seed0从22/300降11/300，seed1从0/300升18/300。

三步训练NLL0.005–0.066对开发2.478–3.841，存在明显训练—开发落差，不能唯一拆分过拟合与A/B→C分布差异。根观测本身也有大量状态错误。C20根均已用于开发，不读D、不采新数据、不运行闭环、不执行162根草案。旧无标签交互预训练和本次带符号监督RSSM相关但不相同，只能作本次两臂内部训练因素比较。

阅读入口：REPORT.md、REPORT.html、RSSM编码器梯度对照_20261019.pdf（5页），3组PNG/PDF图。全字段统计all_metrics.csv，逐根root_metrics.csv，配对差paired_differences.csv。下一轮仅设计20/40/60整根嵌套；探索性组合留出删除16根后余44根，未执行额外拟合。

实验绝对路径：{E}
报告绝对路径：{R}

只读复验：
```bash
source {P}/world_model_env.sh
python {E}/source/verify_sealed.py
```

训练与评估入口保留，但本轮已封存，不应再次启动train.py或finish_pipeline.py覆盖交付。全权重、恢复点、预测、配置与原始日志均留在实验目录；轻量交付包不复制大权重及旧原始图像，哈希和原路径保留。无需额外权限或用户决策即可阅读复核；剩余限制是研究证据边界，不是本轮执行阻塞。原完整HULC/约6000转移/多技能TS-LTL指南仍未完成。
'''
 (R/'HANDOFF.md').write_text(handoff)
 manifest=E/'configs/delivery_manifest.json'
 files={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)}for base in[E,R]for p in sorted(base.rglob('*'))if p.is_file()and p!=manifest}
 save(manifest,{'utc':now(),'scope':'complete experiment and report; historical inputs covered by frozen.json','files':files,'main_model_fits':4,'main_updates_each':6000,'probe_fits':4,'probe_updates_each':600,'no_D_or_new_physics':True})
 for p in[*(Path(x)for x in files),manifest]:p.chmod(0o444)
 print(json.dumps({'files':len(files),'bytes':sum(x['bytes']for x in files.values()),'manifest_sha256':sha(manifest)},ensure_ascii=False))
if __name__=='__main__':main()
