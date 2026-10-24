from common import *
# Coordinates manually transcribed after visually inspecting raw_pages1..5.
# None is an explicit uncertain annotation, not a failed detector or negative label.
DRAWER={0:[53,121,165,158],2:[43,163,139,198],4:[55,122,165,158],6:[40,164,137,200],8:[55,173,151,200],10:[55,169,145,199],12:[59,168,158,195],14:[54,177,150,200],16:[42,166,138,200],18:[45,163,140,197],20:[56,122,164,158],22:[47,162,140,194],24:[56,174,151,200],26:[52,177,149,200],28:[68,131,179,158],30:[56,176,150,200],32:[42,166,139,200],34:[43,168,140,200],36:[58,178,152,200],38:[54,177,150,200]}
FINGERS={
0:[[65,49,77,65],[78,48,91,67]],1:[[2,0,35,28],[49,0,78,26]],
2:[[69,94,86,111],[88,91,102,111]],3:[[2,0,29,27],[56,0,81,26]],
4:[[53,75,66,92],[76,71,88,91]],5:[[1,0,29,27],[55,0,82,26]],
6:[[82,119,98,135],[97,117,111,136]],7:[[11,0,35,27],[50,0,75,25]],
8:[[68,116,81,132],[87,113,99,131]],9:[[8,0,35,26],[49,0,72,25]],
10:[[69,142,82,157],[83,140,96,157]],11:[[8,0,35,27],[49,0,74,25]],
12:None,13:[[6,0,41,33],[42,0,63,32]],
14:[[31,82,43,98],[45,78,56,97]],15:[[7,0,35,24],[49,0,73,26]],
16:[[13,57,23,80]],17:[[2,0,29,28],[55,0,83,27]],
18:[[55,70,67,85],[69,68,82,84]],19:[[11,0,35,27],[49,0,71,26]],
20:[[44,66,56,82],[59,64,72,81]],21:[[11,0,35,27],[50,0,74,26]],
22:None,23:[[6,0,32,27],[52,0,77,26]],
24:[[53,138,66,153],[66,135,83,153]],25:[[3,0,29,27],[55,0,83,26]],
26:[[86,119,102,134],[110,118,121,134]],27:[[4,0,28,27],[55,0,83,27]],
28:[[48,63,62,82],[69,61,82,80]],29:[[1,0,29,27],[55,0,83,26]],
30:[[113,90,125,104],[126,88,140,103]],31:[[14,0,42,34],[42,0,67,33]],
32:None,33:[[2,0,29,27],[55,0,83,27]],
34:[[34,114,49,126],[61,101,70,116]],35:[[2,0,31,25],[52,0,82,26]],
36:None,37:[[2,0,29,26],[55,0,83,26]],
38:[[56,108,67,124],[75,94,87,114]],39:[[1,0,29,27],[55,0,83,27]]}
OCCLUDED_FINGER={0,2,6,8,10,14,18,20,30,38}
def main():
 assert not(E/'data/annotations/round1.json').exists();anns=[]
 for r in read(E/'data/public/images.json'):
  i=r['image_id'];bb=DRAWER.get(i);f=FINGERS[i]
  drawer={'category':'drawer_front','status':'uncertain'if bb is None else'partial'if bb[3]==200 or i==12 else'visible','reason':'not_identifiable'if bb is None else'occluded'if i==12 else'out_of_frame'if bb[3]==200 else'none','boxes':[]if bb is None else[bb],'note':'RGB cannot confidently separate movable front-panel boundary from cavity/table surfaces in this wrist view; no box inferred from other camera'if bb is None else'visible front-panel-plus-handle extent; cavity and drawer side panels excluded'}
  if f is None:status='uncertain';reason='not_identifiable';note='Cannot confidently separate distal fingers from palm/arm or occluding drawer/object at this RGB resolution; do not invent boxes'
  elif r['camera']==1:status='partial';reason='out_of_frame';note='Two visible finger portions clipped at upper image boundary; do not complete unseen palm or finger lengths'
  elif i==16:status='partial';reason='out_of_frame';note='One identifiable finger visible at left frame edge; other finger outside/ambiguous, no hidden box'
  elif i in OCCLUDED_FINGER:status='partial';reason='occluded';note='Visible finger portions adjacent to held object; held object not included'
  else:status='visible';reason='none';note='Separate visible distal finger boxes; not whole end effector'
  finger={'category':'gripper_finger','status':status,'reason':reason,'boxes':[]if f is None else f,'note':note}
  anns.append({'image_id':i,'reviewed_source':f'raw_page_{i//8+1}.png','classes':[drawer,finger]})
 save(E/'data/annotations/round1.json',{'utc':now(),'annotator':'assistant visually inspected each of40RGB images; manual coordinate transcription; not independent human gold','annotations':anns})
 print('MANUAL_RGB_FIRST_PASS_SAVED',len(anns),flush=True)
if __name__=='__main__':main()
