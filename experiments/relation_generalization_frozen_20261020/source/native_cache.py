from common import *
import torch
import features
original=features.make_cache
def native_cache():
 deterministic=torch.are_deterministic_algorithms_enabled()
 # Previous sealed collection used the stock native ROIAlign path without this flag.
 # Torchvision switches to a compiled Python fallback under deterministic CUDA;
 # this independent environment has no compiler. No caching/evaluation had completed.
 torch.use_deterministic_algorithms(False)
 try:original()
 finally:torch.use_deterministic_algorithms(deterministic)
features.make_cache=native_cache
features.main()
save(S/'results/cache_runtime_adjustment.json',{'utc':now(),'original_error':'Torchvision deterministic CUDA ROIAlign selected torch.compile fallback; C compiler unavailable. No cache windows had been written.','adjustment':'Use stock native ROIAlign during FrozenVision caching, as previous sealed cache.py did; restore deterministic algorithms for front encoder and fixed-model inference.','frozen_source_files_unchanged':True,'native_pixels_and_features_not_redefined':True,'training_updates':0,'model_scores_seen_before_adjustment':False,'physical_data_unchanged':True,'original_error_log_retained':True})
