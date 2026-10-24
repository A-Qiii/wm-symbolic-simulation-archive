from common import *
import runpy
before=read(S/'results/analysis.json')
original=json.dumps
def safe_dumps(value,*args,**kwargs):return original(plain(value),*args,**kwargs)
json.dumps=safe_dumps
try:runpy.run_path(str(S/'source/score.py'),run_name='__main__')
finally:json.dumps=original
after=read(S/'results/analysis.json')
before.pop('utc');after.pop('utc');assert before==after
save(S/'results/scoring_execution_receipt.json',{'utc':now(),'original_error':'Final console json.dumps rejected numpy.bool_; all metric files and analysis were already saved successfully.','remedy':'Re-executed unchanged frozen score.py with JSON serializer converting NumPy scalars to native values. No model, data, metric, or threshold change.','all_analysis_values_equal_except_timestamp':True,'original_console_log_retained':True,'frozen_scoring_source_unchanged':True})
