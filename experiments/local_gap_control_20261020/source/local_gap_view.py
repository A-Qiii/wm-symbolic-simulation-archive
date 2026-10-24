import pathlib,json,os,sys,numpy as np
P=pathlib.Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/local_gap_control_20261020';os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.load(open(S/'results/visibility_rows.json'));wanted=['A_role_train_04_Hred_Tpink_b2','A_role_development_01_Hred_Tpink_b2'];selected=[next(r for r in rows if r['episode_id']==n)for n in wanted];selected.append(rows[0])
def project(f,view,point):
 h,w=f['depth_'+view].shape;focal=h/(2*np.tan(np.deg2rad(float(f['calib_'+view+'_fov']))/2));v=np.asarray(f['calib_'+view+'_viewMatrix']).reshape(4,4).T;q=v@np.r_[point,1.];return np.array([focal*q[0]/-q[2]+w//2,-focal*q[1]/-q[2]+h//2])
fig,axs=plt.subplots(len(selected),3,figsize=(12,10));audit=[]
for ri,r in enumerate(selected):
 with np.load(r['public_path'])as z:f={k:z[k]for k in z.files}
 for vi,view in enumerate(['static','gripper']):
  ax=axs[ri,vi];ax.imshow(f['rgb_'+view]);ax.set_axis_off();ax.set_title(r['episode_id']+' / '+view,fontsize=8)
  for gi,g in enumerate(r['gaps']):
   q=g['directions'][3];c=project(f,view,g['center']);v=project(f,view,q['point']);ax.plot([c[0],v[0]],[c[1],v[1]],color=['yellow','cyan'][gi]);ax.scatter(v[0],v[1],marker='x',color=['yellow','cyan'][gi]);ax.text(c[0],c[1],g['color'],color=['yellow','cyan'][gi],fontsize=8)
 ax=axs[ri,2]
 for gi,g in enumerate(r['gaps']):
  lo,hi=np.array(g['visible_bounds']);c=np.array(g['center']);q=g['directions'][3];p=np.array(q['point']);col=['#b39200','#008b9c'][gi];ax.add_patch(plt.Rectangle((lo[1],lo[2]),hi[1]-lo[1],hi[2]-lo[2],fill=False,color=col));ax.plot([hi[1],p[1]],[c[2],p[2]],'-x',color=col,label=g['color']);audit.append({'episode':r['episode_id'],'object':g['color'],'center':c.tolist(),'forward_point':p.tolist(),'gap':q['distance_m'],'support':q['support_points']})
 ax.set_xlabel('world Y (m)');ax.set_ylabel('world Z (m)');ax.legend();ax.set_title('Visible AABB and forward support');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(S/'reports/visibility_projection.png',dpi=160);plt.close(fig);(S/'results/projection_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
