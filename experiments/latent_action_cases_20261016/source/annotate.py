from common import *
def main():
 records={r['pair_id']:r for r in read(REPAIR/'results/D2_records.json')if r['arm']=='canonical_matched'and r['seed']==0};cases=read(E/'results/cases.json')
 notes={
 'A':{'actual_event':'抽屉从关闭变为打开；手中无物，三块均未入屉。','agreement':'一致：持物、三块入屉与抽屉状态全部一致。','mismatch':'本例没有离散命题错误。','diagnostic':'真实后继经同一查询头读取也与真值一致。'},
 'B':{'actual_event':'粉块被抓起；抽屉保持打开，粉块未入屉。','agreement':'一致：持粉块、抽屉打开。','mismatch':'错误：多预测了“粉块入屉”。','diagnostic':'同一查询头能正确读取真实后继；预测表示与真实表示存在语义差异。'},
 'C':{'actual_event':'红块放入抽屉并松手；抽屉保持打开。','agreement':'一致：红块入屉、抽屉打开。','mismatch':'错误：放置后仍预测“持红块”。','diagnostic':'真实后继经查询头读取还漏判红块入屉；读出能力本身也有限。'},
 'D':{'actual_event':'执行器因抽屉关闭拒绝放置；仍持蓝块，抽屉未打开。','agreement':'一致：持蓝块、三块均未入屉。','mismatch':'错误：把实际关闭的抽屉预测成打开。','diagnostic':'同一查询头对真实后继也误判“打开”，不能把此错误全部归因于转移模型。'}}
 for r in cases:
  rec=records[r['id']];r.update(notes[r['case_id']]);r['actual_encoded_readout_probabilities']=rec['actual_successor'];r['actual_encoded_readout_semantics']=semantics(discrete(rec['actual_successor']));r['current_encoded_readout_semantics']=semantics(discrete(rec['current']))
 save(E/'results/cases.json',cases)
 q=REPAIR/'models/query_seed0.pt';src=read(E/'configs/sources.json');src['sha256'][str(q)]=sha(q);save(E/'configs/sources.json',src)
 save(E/'results/readout_caveat.json',{'source':'Saved D2_records.json; no new query inference.','query_checkpoint':str(q),'sha256':sha(q),'case_D':'Actual encoded successor is also classified drawer-open by the same frozen query head. Attribution solely to transition dynamics is unsupported.','no_new_model_selection_or_threshold_change':True})
if __name__=='__main__':main()
