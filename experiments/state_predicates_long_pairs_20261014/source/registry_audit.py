from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def qualifies(rgb,j):
 r,g,b=rgb/255
 return bool(r>.35 and r>g*1.7 and r>b*1.7)if j==0 else bool(b>.35 and b>r*1.7 and b>g*1.7)if j==1 else bool(r>.35 and b>.35 and g<min(r,b)*.6)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);args=ap.parse_args();cache=read(E/'results'/f'cache_{args.phase}.json');meta=read(E/'data/public'/f'windows_{args.phase}.json');by={r['window_id']:r for r in meta};pairs=read(E/'data/public'/f'pairs_{args.phase}.json');details=[]
 for a in cache['audits']:
  for c,q in a['registry_audit'].items():
   if q['match']:continue
   j=C.index(c);r=by[a['window_id']];evidence=[]
   for fi,fp in zip(r['history_frame_indices'],r['public_frames']):
    with np.load(fp)as z:
     for cam in ['static','gripper']:
      mask=z['seg_'+cam];actual=(mask>=0)&((mask&((1<<24)-1))==q['private_uid']);wrong=mask==q['registered_public_uid']if q['registered_public_uid']is not None else np.zeros_like(mask,bool);mean=z['rgb_'+cam][wrong].mean(0)if wrong.any()else None
      evidence.append({'frame':fi,'camera':cam,'true_entity_visible_pixels':int(actual.sum()),'registered_mask_visible_pixels':int(wrong.sum()),'registered_mask_mean_RGB':mean.tolist()if mean is not None else None,'registered_mask_meets_frozen_colour_rule':qualifies(mean,j)if mean is not None else False})
   details.append({'window_id':a['window_id'],'root_id':r['root_id'],'split':r['split'],'color':c,**q,'true_entity_invisible_all_inputs':max(v['true_entity_visible_pixels']for v in evidence)==0,'affected_pairs':[p['pair_id']for p in pairs if a['window_id']in[p['before_window_id'],p['after_window_id']]],'pixel_evidence':evidence})
 wrong_windows={d['window_id']for d in details if not d['missing']};missing_windows={d['window_id']for d in details if d['missing']};bad_windows=wrong_windows|missing_windows;split={}
 for s in ['train','dev']:
  pp=[p for p in pairs if p['split']==s];split[s]={'pairs':len(pp),'pairs_with_present_wrong_slot':sum(p['before_window_id']in wrong_windows or p['after_window_id']in wrong_windows for p in pp),'pairs_with_missing_slot':sum(p['before_window_id']in missing_windows or p['after_window_id']in missing_windows for p in pp),'all_entity_registry_correct_both_endpoints':sum(p['before_window_id']not in bad_windows and p['after_window_id']not in bad_windows for p in pp),'all_entity_registry_correct_and_all7_GT_valid_both_endpoints':sum(p['before_window_id']not in bad_windows and p['after_window_id']not in bad_windows and all(next(a for a in cache['audits']if a['window_id']==p['before_window_id'])['all7_targets']['valid'])and all(next(a for a in cache['audits']if a['window_id']==p['after_window_id'])['all7_targets']['valid'])for p in pp)}
 result={'phase':args.phase,'details':details,'splits':split,'wrong_slot_windows':len(wrong_windows),'missing_slot_windows':len(missing_windows),'no_correction_or_GT_substitution':True,'first_wrong_slot':next((x for x in details if not x['missing']),None)};save(E/'results'/f'registry_diagnostic_{args.phase}.json',result)
 first=result['first_wrong_slot']
 if first:
  r=by[first['window_id']];im=Image.new('RGB',(1180,590),'#f5f8fa');draw=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',13);draw.text((14,10),'原颜色注册错配的因果证据（只读诊断，未修正模型输入）',font=font,fill='#19384b');draw.text((14,37),first['window_id']+' | red真实UID '+str(first['private_uid'])+'，注册UID '+str(first['registered_public_uid']),font=small,fill='#19384b');draw.text((14,61),'上排为实际RGB；下排橙色只标示被原规则登记为红色的可见分割区域。',font=small,fill='#19384b')
  for ci,(fi,fp)in enumerate(zip(r['history_frame_indices'],r['public_frames'])):
   x=14+ci*291;draw.text((x,90),f'frame {fi} / {(fi-r["anchor_frame"])/30:+.1f}s',font=small,fill='#19384b')
   with np.load(fp)as z:
    for cam,dx,sz in [('static',0,196),('gripper',198,84)]:
     rgb=z['rgb_'+cam];mask=z['seg_'+cam]==first['registered_public_uid'];overlay=rgb.copy();overlay[mask]=(rgb[mask]*.35+np.array([244,140,32])*.65).astype(np.uint8);im.paste(Image.fromarray(rgb).resize((sz,sz)),(x+dx,115));im.paste(Image.fromarray(overlay).resize((sz,sz)),(x+dx,335))
  draw.text((14,548),'橙色不是红块真值：红块在本窗双摄全部不可见，背景均值R/G比超过原1.7阈值，触发了红色规则。',font=small,fill='#19384b');im.save(E/'figures'/f'registry_first_wrong_{args.phase}.png')
 print(json.dumps({k:v for k,v in result.items()if k not in ['details','first_wrong_slot']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
