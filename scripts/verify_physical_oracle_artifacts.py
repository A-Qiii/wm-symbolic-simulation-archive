from pathlib import Path
import json,re,hashlib,datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont

P=Path('/home/robot/research/projects/wm_symbolic');B=P/'runs/physical_oracle_v1';R=P/'reports'
def main():
    files=sorted((B/'independent_acceptance').glob('*/*/*/result.json'))
    def check(f):
        r=json.loads(f.read_text());rows=[]
        for cam in ['rgb_static','rgb_gripper']:
            v=f.parent/(cam+'.mp4');frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(v));rows.append({'path':str(v.relative_to(P)),'frames':frames,'expected_frames':r['controls']+1,'passed':frames==r['controls']+1})
        return rows
    checked=[r for batch in ThreadPoolExecutor(max_workers=4).map(check,files) for r in batch]
    assert len(checked)==360 and all(r['passed'] for r in checked)
    (B/'video_integrity.json').write_text(json.dumps({'checked':len(checked),'all_passed':True,'videos':checked},indent=2))
    preview=R/'physical_oracle_v1_previews';preview.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    names={'open_drawer':'打开抽屉','close_drawer':'关闭抽屉','lift_red_block_table':'抓取红块','lift_pink_block_table':'抓取粉块','place_in_drawer':'放入抽屉'}
    clips=[]
    for rid,caption in [('A_acceptance_00','预先固定的首个 A 验收根 · 严格通过'),('C_acceptance_02','唯一有副作用的根 · 链整体判失败')]:
        tasks=sorted((B/'independent_acceptance'/rid/'chain').glob('*/result.json'))
        path=preview/(rid+'_full_chain.mp4');w=imageio.get_writer(str(path),fps=30,codec='libx264',quality=8,macro_block_size=1);count=0
        try:
            for si,f in enumerate(tasks):
                r=json.loads(f.read_text());readers=[imageio.get_reader(str(f.parent/(cam+'.mp4'))) for cam in ['rgb_static','rgb_gripper']]
                try:
                    for fi,(static,gripper) in enumerate(zip(*readers)):
                        if si and fi==0:continue
                        canvas=Image.new('RGB',(800,488),'#f5f7fa');draw=ImageDraw.Draw(canvas)
                        draw.text((16,8),caption,font=font,fill='#b34034' if rid.startswith('C') else '#21694f')
                        draw.text((16,42),f'{si+1}/4 {names[r["skill"]]}  ·  步 {fi}/{r["controls"]}  ·  '+('本技能通过' if r['conservative_pass'] else '抓起目标，但误碰滑门'),font=small,fill='#233346')
                        draw.text((16,70),'静态相机',font=small,fill='#53647a');draw.text((416,70),'手眼相机',font=small,fill='#53647a')
                        canvas.paste(Image.fromarray(static).resize((400,400)),(0,88));canvas.paste(Image.fromarray(gripper).resize((400,400)),(400,88));w.append_data(np.asarray(canvas));count+=1
                finally:
                    for rd in readers:rd.close()
        finally:w.close()
        n,_=imageio_ffmpeg.count_frames_and_secs(str(path));assert n==count
        clips.append({'root':rid,'selection':caption,'path':str(path.relative_to(P)),'frames':count,'full_chain_no_control_frames_omitted':True})
    (preview/'provenance.json').write_text(json.dumps(clips,indent=2,ensure_ascii=False))
    htmlpath=R/'physical_oracle_v1.html';body=htmlpath.read_text();insert='<h2>连续四步双相机回放</h2><p>左侧为固定首个 A 验收根；右侧为发生滑门副作用的 C 根。均保留四步的全部控制帧，只去掉技能边界重复的起始帧。视频上方明确标记技能和通过情况。</p>'+''.join('<video controls preload="metadata" src="'+str(Path(c['path']).relative_to('reports'))+'"></video>' for c in clips)
    body=body.replace('<h2>逐条独立验收与原始视频</h2>',insert+'<h2>逐条独立验收与原始视频</h2>');htmlpath.write_text(body)
    missing=[];links=re.findall(r'(?:href|src)="([^"]+)"',body)
    for x in links:
        if not x.startswith(('http','#')) and not (R/x).resolve().exists():missing.append(x)
    assert not missing,missing
    (B/'report_integrity.json').write_text(json.dumps({'local_links':len(links),'all_local_links_exist':True,'video_frames_verified':len(checked),'previews':clips,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2,ensure_ascii=False))
    print(json.dumps({'videos':len(checked),'previews':clips,'local_links':len(links),'all_passed':True},ensure_ascii=False))
if __name__=='__main__':main()
