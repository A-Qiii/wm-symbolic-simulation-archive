from common import *
import collections,torch
source=S/'source/compute.py'
scope={'__name__':'resume_frozen_diagnostic','__file__':str(source),'collections':collections}
exec(compile(source.read_text(),str(source),'exec'),scope)
def completed_pixel_check():
 result=read(S/'results/fresh_pixel_alignment.json');assert result['passed']and result['fresh_DINO_windows']==5 and result['fresh_spatial_batches']==2
 torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
scope['pixels_and_E']=completed_pixel_check
if __name__=='__main__':
 scope['main']()
 receipt=read(S/'results/compute_complete.json');receipt['timing_scope']='Resumed paired RSSM pass only; pixel/E check completed in first attempt before missing collections import. First-attempt time/peak not captured.';receipt['preserved_failure']='logs/compute_attempt1.log and results/FAILED_attempt1.json';save(S/'results/compute_complete.json',receipt)
