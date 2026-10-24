"""Read-only result interpretation; creates local figures, examples and report."""
from common import *
import csv,html,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image,ImageDraw,ImageFont

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'#f7f9fa','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#f7f9fa'})
ZH=['握持红','握持蓝','握持粉','红在抽屉','蓝在抽屉','粉在抽屉','抽屉打开']
LABEL={'full':'当前状态＋动作','action_only':'仅动作与声明目标','state_only':'仅当前状态','persistence':'保持当前状态','real_next_observation':'真实后继读出（参照）'}
COLOUR={'full':'#087f8c','action_only':'#dc7d32','state_only':'#8660a3','persistence':'#84929c','real_next_observation':'#466d42'}
def esc(x):return html.escape(str(x))
def pct(x):return 'NA'if x is None else f'{100*x:.1f}%'
def table(headers,rows):return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
def image_html(name,caption):return f'<figure><a href="figures/{name}"><img src="figures/{name}" alt="{esc(caption)}"></a><figcaption>{esc(caption)}</figcaption></figure>'
def dumpcsv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def short(name):return LABEL.get(name,LABEL.get(name.split('_seed')[0],name))+(' / seed'+name.rsplit('seed',1)[1]if '_seed'in name else '')
def main():
    ev=read(E/'results/evaluation.json');selection=read(E/'configs/model_selection_frozen.json');checks=read(E/'results/forward_contract_checks.json');validation=read(E/'results/final_validation.json');protocol=read(E/'configs/protocol.json')
    assert validation['passed'];figs=E/'figures';figs.mkdir(exist_ok=True)
    reps=ev['representatives'];runs=selection['all_runs']; modes=['full','action_only','state_only']; modelnames=['persistence']+[reps[m]for m in modes]+['real_next_observation']
    # Training plots use only training batches and the already recorded C-tune scores.
    fig,axes=plt.subplots(1,3,figsize=(16,4.7)); styles=['-','--',':']
    for r in runs:
        curve=[json.loads(q)for q in (E/'runs'/r['run']/'curve.jsonl').read_text().splitlines()]
        for ax,key in [(axes[0],'batch_BCE'),(axes[1],'batch_latent_SmoothL1')]:
            yy=np.array([q[key]for q in curve]);yy=np.convolve(yy,np.ones(20)/20,'valid')
            ax.plot(np.arange(20,601),yy,color=COLOUR[r['mode']],linestyle=styles[r['seed']],lw=1.6,alpha=.85)
        cp=r['all_checkpoints'];axes[2].plot([x['update']for x in cp],[x['tune_root_macro_BCE']for x in cp],marker='o',color=COLOUR[r['mode']],linestyle=styles[r['seed']],lw=1.6)
    for ax,title,ylabel in zip(axes,['训练符号损失','训练隐状态损失','C tune：选模分数'],['加权 BCE（对数轴）','SmoothL1（对数轴）','根级均值 BCE（对数轴）']):
        ax.set_title(title);ax.set_yscale('log');ax.set_ylabel(ylabel);ax.set_xlabel('优化更新次数');ax.grid(alpha=.18)
    handles=[plt.Line2D([0],[0],color=COLOUR[m],label=LABEL[m],lw=2)for m in modes]+[plt.Line2D([0],[0],color='#4c5a64',ls=styles[s],label='seed'+str(s))for s in range(3)]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False);fig.suptitle('固定预算：每种模型 × 3种子 × 600更新',fontsize=16);fig.tight_layout(rect=(0,.09,1,.92));fig.savefig(figs/'training_curves.png',dpi=160);plt.close(fig)
    # Compare root-macro metrics. Error bars denote the full seed range, not confidence intervals.
    metrics=['changed_root_macro_strict_accuracy','unchanged_root_macro_strict_accuracy','all7_pair_root_macro_rate'];displaymetrics=['变化事实','不变事实','全7项同时正确'];bars=['persistence','action_only','state_only','full','real_next_observation']
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),sharey=True)
    for ax,sn,title in zip(axes,['C_all','C_calibration'],['全部C：6根 / 24对','C calibration：3根 / 12对']):
        for j,m in enumerate(bars):
            vals=[];lo=[];hi=[]
            for key in metrics:
                if m in modes:
                    v=ev['three_seed_aggregates'][m][sn][key];mean=v['mean'];vv=v['values_in_seed_order'];low=mean-min(vv);high=max(vv)-mean
                else:
                    q=ev['results'][m][sn]['all'];mean=q['root_macro_all7_pair_rate']if key==metrics[2]else q['mask_metrics']['changed'if key==metrics[0]else'unchanged']['root_macro_strict_accuracy'];low=high=0
                vals.append(mean*100);lo.append(low*100);hi.append(high*100)
            x=np.arange(3)+(j-2)*.15;ax.bar(x,vals,width=.14,color=COLOUR[m],alpha=.9,label=LABEL[m],yerr=[lo,hi],capsize=3,error_kw={'lw':1})
            if m in ['action_only','full']:
                for xx,yy in zip(x,vals):ax.text(xx,yy+2,f'{yy:.0f}',ha='center',fontsize=8)
        ax.set_xticks(np.arange(3),displaymetrics);ax.set_ylim(0,114);ax.set_title(title);ax.set_ylabel('根级平均严格正确率（%）');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False);fig.suptitle('完整模型的增益在于保留其他状态；变化项在“仅动作”中已经饱和',fontsize=15);fig.tight_layout(rect=(0,.1,1,.92));fig.savefig(figs/'C_comparison.png',dpi=160);plt.close(fig)
    # Each row is an independent reset root, with four separately initialized one-step predictions.
    names=['persistence']+[f'{m}_seed{s}'for m in modes for s in range(3)]+['real_next_observation']
    roots=sorted(protocol['data']['dev_roots']); matrix=np.zeros((6,len(names)),int)
    for j,n in enumerate(names):
        rd={r['root_id']:r for r in ev['results'][n]['C_all']['all']['root_details']}
        for i,rid in enumerate(roots):matrix[i,j]=rd[rid]['all7_known_correct_pairs']
    fig,ax=plt.subplots(figsize=(14,5.7));im=ax.imshow(matrix,vmin=0,vmax=4,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(np.arange(len(names)),[short(n).replace(' / ','\n')for n in names],rotation=28,ha='right',fontsize=9)
    ax.set_yticks(range(6),[r.replace('C_longpair_','')+(' / tune'if r in protocol['data']['dev_roles']['tune']else' / cal')for r in roots]);ax.set_ylabel('独立复位根：颜色 / 初始开闭 / 重复号')
    for i in range(6):
        for j in range(len(names)):ax.text(j,i,f'{matrix[i,j]}/4',ha='center',va='center',color='white'if matrix[i,j]>=3 else'#182b3a',fontsize=10)
    fig.colorbar(im,ax=ax,ticks=[0,1,2,3,4],label='每根全7项正确的单步对数');ax.set_title('逐根保留结果；深色表示更多完整状态正确（不是4步连续预测）');fig.tight_layout();fig.savefig(figs/'C_roots.png',dpi=160);plt.close(fig)
    # Latent fidelity is a diagnostic; never used to change checkpoints or budget.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));fullchecks=[r for r in checks if r['run'].startswith('full_')]
    x=np.arange(3)
    for j,(key,label,col)in enumerate([('object_MAE','对象隐状态','#087f8c'),('scene_MAE','场景隐状态','#d5a547')]):
        axes[0].bar(x+(j-.5)*.32,[r['latent_MAE_AB_C']['dev'][key]for r in fullchecks],width=.3,label=label,color=col)
    axes[0].set_ylabel('预测与真实后继隐状态的平均绝对误差');axes[0].legend(frameon=False);axes[0].set_title('完整模型：C隐状态误差')
    axes[1].bar(x,[r['latent_MAE_AB_C']['dev']['predicted_coordinates_outside_GRU_range']*100 for r in fullchecks],width=.5,color='#8660a3');axes[1].set_ylabel('超出冻结GRU取值范围 [-1,1] 的坐标（%）');axes[1].set_title('隐状态范围偏离；越低越好')
    for ax in axes:ax.set_xticks(x,['seed0','seed1','seed2']);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('符号正确不等于隐状态重建准确：不能据此推断可多步滚动',fontsize=15);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(figs/'latent_diagnostics.png',dpi=160);plt.close(fig)
    # Full machine-readable exports include all models, roots, masks and diagnostic subsets.
    rows=[];group_rows=[];pair_rows=[]
    for name,sets in ev['results'].items():
        for sn,subsets in sets.items():
            for sub,r in subsets.items():
                for mask,q in r['mask_metrics'].items():rows.append({'model':name,'set':sn,'subset':sub,'pairs':r['pairs'],'roots':r['roots'],'all7_known_correct_pairs':r['all7_known_correct_pairs'],'all_pairs_correct_roots':r['all_pairs_correct_roots'],'mask':mask,**q})
                for group,q in r['predicate_groups'].items():group_rows.append({'model':name,'set':sn,'subset':sub,'predicate_group':group,**q})
    for name,detail in ev['details'].items():
        for r in detail:
            for j in range(7):pair_rows.append({'model':name,'pair_id':r['pair_id'],'root_id':r['root_id'],'split':r['split'],'dev_role':r['dev_role'],'skill':r['skill'],'predicate':NAMES[j],'probability':r['probabilities'][j],'truth':r['truth'][j],'valid':r['valid'][j],'UNKNOWN':j in r['UNKNOWN_columns'],'raw_error':j in r['raw_error_columns'],'changed':j in r['changed_columns'],'target_entity':j in r['target_entity_columns'],'all7_known_correct':r['all7_known_correct'],'current_reader_wrong_or_UNKNOWN':r['current_reader_wrong_or_UNKNOWN'],'missing_at_either_endpoint':r['missing_at_either_endpoint']})
    dumpcsv(E/'results/all_metrics.csv',rows);dumpcsv(E/'results/predicate_groups.csv',group_rows);dumpcsv(E/'results/all_pair_predicates.csv',pair_rows)
    dumpcsv(E/'results/run_budget.csv',[{k:r[k]for k in ['run','seed','mode','updates','parameters','wall_seconds','CUDA_elapsed_seconds','peak_torch_allocated_bytes','peak_torch_reserved_bytes']}|{'selected_update':r['selected']['update'],'tune_score':r['selected']['tune_root_macro_BCE']}for r in runs])
    # Five transparently selected examples; no filtering or new inference is involved.
    pairs=read(E/'data/public/pairs.json');windows={r['window_id']:r for r in read(DATA/'data/public/windows_v1.json')};targets=dict(np.load(E/'data/_private/training_targets_and_audit.npz'));pred=dict(np.load(E/'results/predictions.npz'))
    ci=[i for i,r in enumerate(pairs)if r['split']=='dev'];full=ev['details'][reps['full']]
    def first(cond):return next(i for i in ci if cond(i))
    chosen=[(first(lambda i:pairs[i]['skill']=='open_drawer'and not targets['before_y'][i,3:6].any()),'同为打开抽屉：之前没有物体在内','只看动作的模型无法读取当前抽屉内容。'),(first(lambda i:pairs[i]['skill']=='open_drawer'and targets['before_y'][i,3:6].any()),'同为打开抽屉：之前已有物体在内','动作相同，已有物体的归属必须保留。'),(first(lambda i:pairs[i]['skill'].startswith('lift_')and not ev['details'][reps['state_only']][i]['all7_known_correct']),'拿起物体：需要动作指定的目标','不提供动作时，握持判断可能冲突并产生UNKNOWN。'),(first(lambda i:pairs[i]['skill']=='place_in_drawer'and not ev['details'][reps['action_only']][i]['all7_known_correct']),'放入抽屉：不要同时改变其他物体','变化事实可以全对，但误改非目标事实仍使整状态错误。'),(next(i for i,r in enumerate(full)if r['current_reader_wrong_or_UNKNOWN']),'保留当前读出已出错的样本（AB训练集）','该例不是C泛化证据；当前读出误报握持粉，仍计入全部指标。')]
    fonts={n:ImageFont.truetype(FONT,n)for n in [16,18,20,22,26,30]}
    case_records=[]
    def decoded(p):
        k=known(p[None],[.5,.5,.5])[0];return [('是'if p[j]>=.5 else'否')if k[j]else'未知'for j in range(7)]
    def wrap(draw,text,xy,font,width,fill,dy):
        xx,yy=xy;line=''
        for ch in text:
            if draw.textlength(line+ch,font=font)>width:draw.text((xx,yy),line,font=font,fill=fill);yy+=dy;line=ch
            else:line+=ch
        if line:draw.text((xx,yy),line,font=font,fill=fill)
        return yy+dy
    for num,(i,title,note)in enumerate(chosen,1):
        r=pairs[i];canvas=Image.new('RGB',(1880,1130),'#f4f7f9');d=ImageDraw.Draw(canvas);d.text((35,24),f'{num:02d}  {title}',font=fonts[30],fill='#143346');d.text((35,72),r['pair_id']+'  |  '+str(r['dev_role']or'train'),font=fonts[20],fill='#466273');d.text((35,108),'声明动作：'+r['skill']+' '+str(r['declared_arguments'])+'；输出为符号后果，不生成未来图像。',font=fonts[22],fill='#143346')
        for xx,head in [(35,'实际执行前图像'),(650,'当前隐状态＋动作 → 预测后果'),(1260,'实际执行后图像')]:d.text((xx,165),head,font=fonts[26],fill='#143346')
        for xx,wid in [(35,r['before_window_id']),(1260,r['after_window_id'])]:
            z=np.load(windows[wid]['public_frames'][-1]);a=Image.fromarray(z['rgb_static']);b=Image.fromarray(z['rgb_gripper']);canvas.paste(a.resize((340,340)),(xx,215));canvas.paste(b.resize((190,190)),(xx+350,215));d.text((xx,560),'固定相机：历史窗口末帧',font=fonts[18],fill='#466273');d.text((xx+350,416),'腕部相机',font=fonts[18],fill='#466273')
        cols=[(35,targets['observed_current_probabilities'][i],targets['before_y'][i],'当前读出与当前GT'),(650,pred[reps['full']][i],targets['after_y'][i],'预测读出与后继GT'),(1260,pred['real_next_observation'][i],targets['after_y'][i],'真实后继读出与后继GT')]
        for xx,pp,yy,head in cols:
            top=620 if xx!=650 else 230;d.text((xx,top),head+('（seed2）'if xx==650 else''),font=fonts[22],fill='#143346')
            for dx,label in [(8,'事实'),(225,'P(真)'),(360,'读出'),(460,'GT')]:d.text((xx+dx,top+37),label,font=fonts[18],fill='#466273')
            de=decoded(pp)
            for j in range(7):
                y0=top+76+j*43;bad=(pp[j]>=.5)!=bool(yy[j]);colour='#687985'if de[j]=='未知'else'#c66426'if bad else'#087f8c';d.rounded_rectangle((xx,y0-3,xx+540,y0+35),radius=6,fill='#ffffff');d.text((xx+8,y0),ZH[j],font=fonts[20],fill=colour);d.text((xx+225,y0),f'{pp[j]:.3f}',font=fonts[20],fill=colour);d.text((xx+360,y0),de[j],font=fonts[20],fill=colour);d.text((xx+460,y0),'是'if yy[j]else'否',font=fonts[20],fill='#143346')
        yy=660
        for name in ['persistence',reps['action_only'],reps['state_only']]:
            dd=ev['details'][name][i];wrong=[ZH[j]for j in sorted(set(dd['raw_error_columns'])|set(dd['UNKNOWN_columns']))];msg=short(name)+'：'+('全7项正确'if not wrong else'错误/未知：'+'、'.join(wrong));yy=wrap(d,msg,(650,yy),fonts[20],540,'#466273',29)+15
        wrap(d,note,(35,1040),fonts[22],1770,'#143346',32)
        d.text((35,1090),'青色=已知且正确；橙色=错误；灰色=UNKNOWN。概率保留3位小数。红/蓝/粉指真实物体身份。GT只作标签与评分。',font=fonts[18],fill='#466273')
        name=f'case_{num:02d}.png';canvas.save(figs/name);case_records.append({'number':num,'pair_id':r['pair_id'],'title':title,'note':note,'figure':'figures/'+name,'selection':'first matching in frozen pair order; illustrative only','before_window_id':r['before_window_id'],'after_window_id':r['after_window_id']})
    save(E/'results/illustrative_cases.json',case_records)
    # Chinese report keeps headline claims next to their scope and denominators.
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单步隐状态转移：固定预算开发结果</title><style>body{max-width:1260px;margin:36px auto;padding:0 24px;font:16px/1.75 "Noto Sans CJK SC",sans-serif;color:#173446;background:#f7f9fa}h1{font-size:30px}h2{margin-top:44px;border-bottom:2px solid #cad8de;padding-bottom:8px}h3{margin-top:26px}p{max-width:1120px}.callout{background:#e6f2f2;border-left:5px solid #087f8c;padding:18px 24px}.warn{background:#fff1de;border-left:5px solid #dc7d32;padding:18px 24px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dce5e9}th{background:#e5edf1;white-space:nowrap}tr:nth-child(even){background:#f6fafb}figure{margin:24px 0}figure img{max-width:100%;height:auto;border:1px solid #dce5e9}figcaption{font-size:14px;color:#496573;margin:8px 4px}a{color:#087f8c}code{font:13px monospace;overflow-wrap:anywhere}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}.small{font-size:13px;color:#496573}</style><body>']
    parts+=['<h1>单步隐状态转移：固定预算开发结果</h1><p class="small">2026-10-14 · 本机 RTX 5080 Laptop · RGB v1、历史窗口、编码器、共享符号读出器均冻结</p>', '<div class="callout"><b>已证明的范围：</b>在现有A/B训练96对、C开发24对中，完整模型三个种子均达到C全7项24/24正确。它的主要增益是保留不变及非目标事实。每次预测均从真实当前观测开始；这不是多步滚动或独立D验收。</div>', '<div class="warn"><b>预注册决策：暂不进入一次新的独立测试。</b>“仅动作”对照也达到变化事实100%正确，完整模型未满足“变化项严格优于每个对照”的预注册条件。规则保持不变，预算到此结束。先定义能区分状态依赖和失败情况的独立挑战协议，再另行决定是否执行。</div>']
    parts+=['<h2>实验到底输入、预测和比较了什么</h2>',table(['元素','本次实现','为什么这样设置'],[['当前观测','双相机RGB；4个真实历史帧覆盖2.1秒；RGB v1注册固定','只利用动作执行前可获得的信息；不复制末帧充当历史'],['当前隐状态','3个对象各96维＋场景96维；编码器与GRU固定','提供物体、场景以及已有接触/遮挡历史的表示'],['动作与目标','9维声明：4种动作、仅lift具备的3色目标、预算与停留时长','place/open/close没有颜色参数；不把真实持有对象或结果回填到动作'],['可训练转移','76224参数；对象共享残差MLP、场景残差MLP','仅学习当前隐状态＋动作到下一隐状态的映射'],['共享符号读出','原held/in/open头保持权重和阈值；输出7个布尔事实','预测后继与实际后继使用同一读出规则；没有新增8类位置头'],['真实后继隐状态','仅作为分离的监督目标及评估参照','绝不作为转移forward输入；GT也只用于标签与评分'],['仅动作对照','与完整模型同大小；将当前对象和场景输入及残差基底清零','检验只凭技能名称和声明目标能做到什么'],['仅状态对照','同大小；将全部9维动作输入清零','检验是否真正需要动作；保留当前状态输入'],['保持当前状态','直接复用当前读出概率','检验预测变化是否优于什么也不改变']]),'<p>7项事实依次为：握持红、握持蓝、握持粉、红在抽屉、蓝在抽屉、粉在抽屉、抽屉打开。held沿用双指接触且无外物承托的标签；in沿用主支撑/归属且held优先的标签；open沿用原阈值和有效性规则。阈值0.5保持不变；多个held同时为真时，将这些held项标为UNKNOWN。严格正确率把UNKNOWN计作不正确。</p>']
    parts+=['<h2>数据与选择规则</h2>',table(['集合','独立根','转移对','用途'],[['A/B train','24','96','优化模型；类权重仅从这些后继标签计算'],['C tune','3','12','200/400/600更新点，按根平均加权BCE选每种子的检查点'],['C calibration','3','12','训练循环不评分；检查点与代表种子冻结后再评估'],['C合计','6','24','完整开发集报告；此前用于输入诊断，因此不是盲测']]),'<p>同一复位根的所有4次调用始终处于同一划分。所有120对、840个有效事实均保留：150项变化、690项不变。现有数据中120次动作均成功，缺少失败、拒绝执行以及整状态完全不变的反例。缺失观测和当前读出错误只作为诊断子集，未用于剔除。</p><p>固定3个种子（0/1/2），每模型每种子600次更新，批大小16，总计5400次。AdamW：学习率0.001，weight decay 0.0001，梯度裁剪1。符号损失为held/in/open三组等权BCE，另加0.05×对象与场景SmoothL1均值。每种子在各对照间使用相同初始参数和采样顺序。每种子取最低C tune分数检查点，平分取更早；各模型代表种子取最低分，平分取较小种子。C calibration不参与这些选择。</p>']
    parts+=['<h2>结果：完整状态比“变化项命中”更有区分力</h2>']
    for sn,title in [('C_all','C全部：24对 / 6根'),('C_calibration','C calibration：12对 / 3根')]:
        tab=[]
        for m in bars:
            if m in modes:
                a=ev['three_seed_aggregates'][m][sn];vs=a['all7_pair_root_macro_rate']['values_in_seed_order'];paircount=ev['results'][m+'_seed0'][sn]['all']['pairs'];ntext=' / '.join(str(round(v*paircount))for v in vs)+f'（各/{paircount}）';q=[pct(a[k]['mean'])for k in [metrics[0],metrics[1],'target_entity_root_macro_strict_accuracy','other_entity_root_macro_strict_accuracy']];u=' / '.join(str(v)for v in a['UNKNOWN_facts']['values_in_seed_order'])
            else:
                a=ev['results'][m][sn]['all'];ntext=f"{a['all7_known_correct_pairs']}/{a['pairs']}";q=[pct(a['mask_metrics'][k]['root_macro_strict_accuracy'])for k in ['changed','unchanged','target_entity','other_entity']];u=str(a['mask_metrics']['all7']['UNKNOWN'])
            tab.append([LABEL[m],ntext,*q,u])
        parts+=['<h3>'+title+'</h3>',table(['方法','全7项正确对数：seed0/1/2','变化事实','不变事实','目标实体','其他实体','UNKNOWN事实数：seed0/1/2'],tab)]
    parts+=['<p>上表事实比例为各根先计算、再按根等权、最后对三个种子取平均。全7项要求同一对的7项均已知且正确；UNKNOWN列是事实数量而非转移数量。目标实体在lift为声明颜色、place为根任务的计划颜色（仅用于评分），开关抽屉时为抽屉；其余事实属其他实体。完整对象分组、抽屉单独分组与FP/FN见CSV。真实后继读出访问执行后的图像，只是读出器参照，不能作为提前预测的基线能力。</p>',image_html('C_comparison.png','横轴是变化、不变和全7项三种条件；纵轴为根级平均严格正确率。青色完整模型、橙色仅动作、紫色仅状态、灰色保持当前、绿色真实后继参照。误差线是三个种子的最小到最大范围，不是统计置信区间。变化项单独计分会漏掉非目标误改。'),image_html('C_roots.png','每行是一个独立复位根，颜色由浅到深表示该根4次单步调用中有0至4对全7项正确。每列是方法/种子。完整模型各根均4/4，但四次均重新使用真实当前观测，不代表4步自回归预测成功。')]
    perrun=[]
    for n in names:
        q=[ev['results'][n][sn]['all']for sn in ['AB_all','C_tune','C_calibration','C_all']];perrun.append([short(n),*[f"{r['all7_known_correct_pairs']}/{r['pairs']}"for r in q],f"{q[-1]['all_pairs_correct_roots']}/{q[-1]['roots']}"])
    parts+=['<details><summary>展开全部种子的AB/C完整状态计数</summary>',table(['方法','AB','C tune','C calibration','C全部','C根内4对全正确'],perrun),'</details>']
    subsetrows=[]
    for sn in ['AB_all','C_all']:
        for sub in ['current_reader_wrong_or_UNKNOWN','missing_at_either_endpoint']:
            for n in modelnames:
                r=ev['results'][n][sn][sub];subsetrows.append([sn,sub,short(n),r['pairs'],r['roots'],f"{r['all7_known_correct_pairs']}/{r['pairs']}"if r['pairs']else'NA（0样本）',r['mask_metrics']['all7']['UNKNOWN']if r['pairs']else'NA'])
    parts+=['<h2>错误与遮挡子集没有被隐藏</h2><p>当前读出错误的转移只有1对，位于AB训练集：A_longpair_blue_open_1__m0，执行前误报握持粉（P=0.594）。完整模型和仅动作代表在该例后继全7项正确；仅状态代表失败并出现2个UNKNOWN。C该子集为0对，无法估计修复能力。下表也单独报告至少一个端点存在整窗对象不可见的情况，主指标仍包含这些样本。</p>',table(['集合','诊断子集','方法（代表种子）','对数','根数','全7项正确','UNKNOWN'],subsetrows)]
    parts+=['<h2>训练曲线、计算预算与隐状态保真度</h2>',image_html('training_curves.png','左/中：20次更新滑动平均的训练BCE和SmoothL1；右：仅C tune在200/400/600更新时的选模BCE。横轴为更新次数，纵轴均为对数损失、越低越好。颜色对应模型，实线/虚线/点线分别为seed0/1/2。训练总损失=BCE＋0.05×SmoothL1；曲线低不等同独立验收通过。')]
    budget=[]
    for r in runs:budget.append([r['run'],r['updates'],r['selected']['update'],f"{r['selected']['tune_root_macro_BCE']:.6f}",f"{r['wall_seconds']:.3f}",f"{r['CUDA_elapsed_seconds']:.3f}",f"{r['peak_torch_allocated_bytes']/2**20:.2f}",f"{r['peak_torch_reserved_bytes']/2**20:.2f}"])
    parts+=[table(['运行','总更新','选中更新','C tune BCE','墙钟秒','CUDA计时秒','峰值allocated MiB','峰值reserved MiB'],budget),f'<p>九次训练及其循环内检查点/日志合计墙钟{selection["total_wall_seconds"]:.3f}秒；每次计时见表。小型MLP在预缓存隐状态上训练，因此这些数值不包含此前仿真采集、DINO特征缓存、环境启动、模型加载或最终评估。CUDA计时从训练循环开始到结束，包含中间检查点工作造成的间隔；不是独立算子耗时。PyTorch峰值allocated约21.70 MiB、reserved 26 MiB，仅为该进程PyTorch分配器统计，不是整卡占用，更不是完整世界模型训练成本。运行环境：Python3.11.13、PyTorch2.11.0+cu128、RTX5080 Laptop；单GPU串行。</p>',image_html('latent_diagnostics.png','只看完整模型，横轴为三个训练种子。左图青色为对象、金色为场景的C后继隐状态平均绝对误差；右图紫色为超出冻结GRU原始[-1,1]范围的坐标百分比。两项都越低越好。seed1符号同样全对但对象隐状态误差明显更高，说明固定读出头的正确输出不足以证明隐状态动力学可滚动。')]
    latentrows=[]
    for r in checks:
        q=r['latent_MAE_AB_C']['dev'];latentrows.append([r['run'],f"{q['object_MAE']:.4f}",f"{q['scene_MAE']:.4f}",pct(q['predicted_coordinates_outside_GRU_range'])])
    parts+=['<details><summary>展开所有模型的C隐状态诊断</summary>',table(['运行','对象MAE','场景MAE','超范围坐标'],latentrows),'</details><p>代表完整模型seed2在C的对象MAE为0.1422、场景MAE为0.0529，33.0%的预测坐标超出冻结GRU范围。没有据此事后改变结构、加大隐状态损失或继续训练。本次只能说明固定读出头下的单步符号结果；隐状态可用性和多步误差累积仍未通过验证。</p>']
    parts+=['<h2>实际图像 → 预测符号后果 → 实际后果</h2><p>以下5例按固定数据顺序选择各类首例，完整选择记录在illustrative_cases.json。图中真实RGB未生成或改写；展示的是历史窗口末帧，模型实际输入仍为4帧。中央预测区仅为符号读出和概率；GT与右侧后继图像只用于解释和评分。示例不改变总体样本和指标。</p>']
    for c in case_records:parts.append(image_html(Path(c['figure']).name,c['title']+'。'+c['note']))
    parts+=['<h2>预注册决策与下一次测试的条件</h2>',table(['训练前冻结的要求','本次结果','决策依据'],[['C calibration各谓词召回≥95%、FPR≤1%，UNKNOWN≤5%','满足','代表完整模型各组召回100%，FPR=0，UNKNOWN=0'],['三个种子平均变化准确率严格超过所有简单对照','不满足','完整=100%；仅动作=100%，没有严格增益'],['三个种子平均全7项正确率严格超过所有简单对照','满足','完整100%；仅动作33.3%；仅状态41.7%；保持当前0%'],['不变事实不弱于最强简单对照','满足','完整100%；保持当前100%']]),'<p>这项保守门槛在当前“全部成功”的数据里对变化项出现饱和，不能充分区分误改其他事实的模型。完整模型的非目标保护增益是真实开发结果；同时，不能在看到结果后改写门槛并宣布通过。依冻结协议，本次不推荐立即开展同类独立测试，停止在现有预算。</p><p>如以后继续，先冻结一个更有区分力的挑战定义：相同动作但当前状态不同、非目标事实保留、动作失败/拒绝及完全不变情形，并把完整状态与隐状态保真度列入主要指标。这是未来协议建议，本轮未生成新数据、未运行D、未执行闭环或1/3/5步验证，也未上传。</p>']
    parts+=['<h2>复核与本机文件</h2>',table(['检查','结果与证据'],[['不可变输入','20项冻结文件哈希核对；上一版RGB交付939文件 / 78,994,186字节完整核对'],['数据对齐','当前隐状态逐项等于before索引；真实next只在私有目标文件；声明动作逐项重建；GT与上一版一致'],['预算与选择','9次×600=5400更新；27检查点哈希正确；同种子各模型初始权重一致；从C tune记录独立复算选模'],['指标独立复算','660个统计单元重新计数；120对/840事实全保留；全部根划分互斥'],['共享读出器','冻结候选摘要未变；原CUDA路径重放概率最大误差0'],['CPU数值诊断','CPU与CUDA概率最大差0.0002494，阈值标签差0；最初1e-5的CPU容差检查失败记录保留，随后按原CUDA路径精确验证，未改模型或输出']]),'<p>训练前协议、训练源码、检查点与逐更新日志均已保存。最终manifest列出交付文件SHA256；封存后文件只读。初始CPU审计失败及后续验证日志均保留，便于追溯。</p><ul>'+''.join(f'<li><a href="{p}">{label}</a></li>'for p,label in [('configs/protocol.json','训练前冻结协议（含源数据与源码哈希）'),('configs/model_selection_frozen.json','冻结选模记录与27个检查点'),('results/evaluation.json','完整指标与逐对结果JSON'),('results/all_metrics.csv','全部模型/划分/子集/掩码指标CSV'),('results/predicate_groups.csv','held / in / open分组指标CSV'),('results/all_pair_predicates.csv','全部120对×11方法×7谓词的概率与真值CSV'),('results/run_budget.csv','计算预算CSV'),('results/final_validation.json','最终核验'),('results/decision.json','预注册决策'),('delivery_manifest.json','交付哈希清单')])+'</ul><p class="small">此交付完成本次受限单步开发实验；不表示原世界模型—TS/LTL方案或独立测试验收已完成。</p></body></html>']
    (E/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    summary={'created_utc':now(),'scope':'bounded one-step development only','C_all_all7_by_seed':{m:[ev['results'][f'{m}_seed{s}']['C_all']['all']['all7_known_correct_pairs']for s in range(3)]for m in modes},'C_pairs':24,'C_roots':6,'C_calibration_all7_by_seed':{m:[ev['results'][f'{m}_seed{s}']['C_calibration']['all']['all7_known_correct_pairs']for s in range(3)]for m in modes},'C_calibration_pairs':12,'representatives':reps,'training_updates':5400,'total_training_loop_seconds':selection['total_wall_seconds'],'peak_torch_allocated_MiB':max(r['peak_torch_allocated_bytes']for r in runs)/2**20,'decision':ev['decision'],'case_count':len(case_records),'report':'report.html','validation':'results/final_validation.json'}
    save(E/'results/delivery_summary.json',summary);print(json.dumps({'report':str(E/'report.html'),'figures':len(list(figs.glob('*.png'))),'case_count':len(case_records),'csv_rows':{'all_metrics':len(rows),'predicate_groups':len(group_rows),'all_pair_predicates':len(pair_rows)}},ensure_ascii=False))

if __name__=='__main__':main()
