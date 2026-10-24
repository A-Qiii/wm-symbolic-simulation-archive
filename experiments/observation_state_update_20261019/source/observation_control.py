from study import *
from model import ObservationStateWorldModel,OriginalWorldModel
from evaluate import probabilities

def main():
    guard('evaluate');lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');rows=[]
    for seed in [0,1]:
      for variant,cls,folder,arm in [('control',OriginalWorldModel,C,'paired'),('updated',ObservationStateWorldModel,U,'state_update')]:
        selected=read(folder/f'results/train_{arm}_seed{seed}.json')['selected'];m=cls(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(selected['path'],map_location='cuda',weights_only=False)['model']);m.eval().requires_grad_(False);before=statehash(m)
        for split in ['train','development','evaluation']:
          data=NewEpisodes(split)
          for rid in data.root_ids:
            ids=[next(i for i,e in enumerate(data.episodes)if e['root_id']==rid and e['branch']==b)for b in [1,2]];ep=data.episodes[ids[0]];rootseed=int(ep['root']['seed']);target=COLORS.index(ep['root']['matched_color']);held=COLORS.index(ep['root']['held_color'])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
              batches=[data.batch([i],'cuda')for i in ids];enc=[m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(8,0)for b in batches];actions=[m.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(8,0)for b in batches]
              assert torch.equal(enc[0][:,:2],enc[1][:,:2]) and torch.equal(actions[0],actions[1])
              torch.manual_seed(81831+rootseed);state,_=m.initial(enc[0][:,0]);pred,_=m.imagine(state,actions[0][:,0]);state,_=m.observe(pred,enc[0][:,1]);prior,_=m.imagine(state,actions[0][:,1])
              def follow(observation):
                torch.manual_seed(777000+rootseed);state,q=m.observe(prior,observation);h=state.h.float().cpu().numpy();qq=q.float().cpu().numpy();pp=[probabilities(m,state)]
                for t in [2,3,4]:state,_=m.imagine(state,actions[0][:,t]);pp.append(probabilities(m,state))
                return np.array(pp),h,qq
              actual=[follow(e[:,2])for e in enc];same_success=[follow(enc[0][:,2]),follow(enc[0][:,2])];same_failure=[follow(enc[1][:,2]),follow(enc[1][:,2])]
            truth_pair=[truth(b['cat'][0,2:6].cpu().numpy())for b in batches]
            def scores(outs):
              ok=[correct(rel(outs[i][0]),truth_pair[i])[:,3]for i in [0,1]]
              return bool(ok[0][0]and ok[1][0]),bool(ok[0].all()and ok[1].all())
            act_h2,act_joint=scores(actual)
            for name,outs in [('same_success_E',same_success),('same_failure_E',same_failure)]:
              assert all(np.array_equal(x,y)for x,y in zip(outs[0],outs[1]));h2,joint=scores(outs);assert not h2 and not joint
              rows.append({'variant':variant,'seed':seed,'split':split,'root_id':rid,'control':name,'actual_h2_pair_joint':act_h2,'actual_h2_and_suffix3_pair_joint':act_joint,'same_E_h2_pair_joint':h2,'same_E_h2_and_suffix3_pair_joint':joint,'same_E_h_q_probabilities_exact':True,'actual_E_pair_h_rms_difference':float(np.sqrt(np.mean((actual[0][1]-actual[1][1])**2))),'actual_E_pair_q_mean_absolute_difference':float(np.abs(actual[0][2]-actual[1][2]).mean()),'actual_E_target_Held_gap':float(rel(actual[0][0])[0,target]-rel(actual[1][0])[0,target]),'same_E_target_Held_gap':0.,'actual_A_retained_both':bool(rel(actual[0][0])[0,3+held]>=.5 and rel(actual[1][0])[0,3+held]>=.5)})
        assert statehash(m)==before;del m;torch.cuda.empty_cache()
    csvwrite(U/'results/same_observation_control.csv',rows);save(U/'results/same_observation_control.json',{'utc':now(),'rows':len(rows),'all_same_E_loses_pair_discrimination':True,'same_actual_prefix_and_common_random_numbers':True,'actual_E_includes_RGB_features_and_robot':True,'no_h_z_splicing':True,'no_parameter_update':True,'separate_fixed_RNG':'777000+root seed for observation/suffix; distinct from main evaluator, so actual-pair counts are a separate diagnostic, not replacement primary scores'});print('OBSERVATION_CONTROL_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
