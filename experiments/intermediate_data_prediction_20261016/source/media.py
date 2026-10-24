from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from PIL import Image
import subprocess,imageio_ffmpeg
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
def main():
 rows=read(E/'data/ABC/public/states.json');labels=read(E/'data/ABC/_private/labels.json');pred=read(E/'results/true_state_probabilities_C_validation.json');lookup={i:j for j,i in enumerate(pred['state_indices'])};fig,axs=plt.subplots(3,3,figsize=(16,13));manifest=[];(R/'videos').mkdir(exist_ok=True)
 for row,col in enumerate(COLORS):
  root=f'C_intermediate_v3_{col}_2';chosen=[]
  for column,(tag,title)in enumerate([('pre_grasp','抓取前'),('hover_above_drawer','悬在抽屉上方'),('post_place','放置调用后稳定等待')]):
   i=next(i for i,r in enumerate(rows)if r['root_id']==root and tag in r['strata']);r=rows[i];chosen.append(r);z=np.load(r['path']);left=z['rgb_static'];right=np.array(Image.fromarray(z['rgb_gripper']).resize((200,200),Image.Resampling.NEAREST));rgb=np.concatenate([left,right],1);ax=axs[row,column];ax.imshow(rgb);ax.set_xticks([]);ax.set_yticks([]);y=labels[str(i)]['y'];p=discrete(pred['probabilities']['query_new_coverage_seed0'][lookup[i]]);j=COLORS.index(col);drawer=lambda t:['关','开','中间'][int(np.argmax(t[6:]))];ax.set_title(f'{["红","蓝","粉"][row]}块 · {title} · 帧{r["frame"]}',fontsize=11);ax.set_xlabel(f'真值：持物{int(y[j])} / 入屉{int(y[3+j])} / 抽屉{drawer(y)}\n新覆盖头0：持物{int(p[j])} / 入屉{int(p[3+j])} / 抽屉{drawer(p)}',fontsize=10)
   manifest.append({'root_id':root,'stratum':tag,'frame':r['frame'],'public_path':r['path'],'sha256':sha(r['path']),'truth9':y,'query_new_seed0_prediction9':p})
  video=Path(chosen[0]['path']).parent.parent/'continuous.mp4';start=max(0,chosen[1]['frame']/30-1);duration=(chosen[2]['frame']/30+1)-start;output=R/'videos'/f'{root}_hover_to_after_place.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-y','-i',str(video),'-ss',str(start),'-t',str(duration),'-an','-c:v','libx264','-crf','20','-movflags','+faststart',str(output)],check=True);manifest.append({'root_id':root,'video':str(output),'source_video':str(video),'source_sha256':sha(video),'start_seconds':start,'duration_seconds':duration,'reencoded_real_frames_only':True})
 fig.suptitle('真实中间态：新覆盖改善持物读出，也可能损伤抽屉判别',fontsize=16);fig.text(.04,.022,'每格左为固定相机，右为腕相机；使用原RGB颜色，腕图仅最近邻放大，无生成或补帧。每行来自同一C验证根的连续真实交互。\n文字仅突出目标块的持物/入屉与抽屉三类；完整九项真值与预测另存。红块放置后最高角点高出抽屉沿约10.4毫米：虽有承重接触，严格入屉真值仍为0。',fontsize=10);fig.subplots_adjust(left=.035,right=.985,bottom=.14,top=.9,hspace=.55,wspace=.07);fig.savefig(R/'real_intermediate_states.png',dpi=160);plt.close(fig);save(R/'media_sources.json',manifest);print('REAL_MEDIA_READY',flush=True)
if __name__=='__main__':main()
