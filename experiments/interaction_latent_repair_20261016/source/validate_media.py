from common import *
import imageio.v2 as imageio
def main():
 rows=[]
 for r in read(E/'results/D2_collection.json'):rows.append((Path(r['folder']),r['frames']))
 for r in read(E/'results/closed_loop.json'):rows.append((Path(r['video']).parent,r['frames']))
 out=[];counts={'preparation':0,'task':0}
 for i,(folder,n)in enumerate(rows):
  reader=imageio.get_reader(folder/'continuous.mp4');meta=reader.get_meta_data();actual=reader.count_frames();reader.close();assert actual==n and meta['fps']==30,(folder,actual,n,meta['fps'])
  for call in read(folder/'_private/calls.json'):
   assert call['end_frame']-call['primitive_end']==96;assert call['primitive_end']-call['start_frame']==call['native']['controls']<=900;counts[call['role']]+=1
   if call['skill']=='place_in_drawer':assert call['executor_arguments']=={'skill':'place_in_drawer'}and not any(call['action'][4:7])
  out.append({'path':str(folder/'continuous.mp4'),'frames':actual,'fps':meta['fps'],'sha256':sha(folder/'continuous.mp4')})
  if i%12==0:print('VIDEO_VALIDATED',i+1,'/',len(rows),flush=True)
 assert len(out)==84 and counts=={'preparation':24,'task':217};save(E/'results/media_validation.json',{'passed':True,'videos':84,'all_frames':sum(r['frames']for r in out),'native_call_roles':counts,'all_primitive_controls_at_most900':True,'all_postholds96':True,'all_place_contracts_unparameterized':True,'records':out});print('ALL_MEDIA_PASS',84,counts,flush=True)
if __name__=='__main__':main()
