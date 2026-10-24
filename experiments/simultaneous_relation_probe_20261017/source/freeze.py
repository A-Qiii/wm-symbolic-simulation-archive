from config import *
import ast,shutil,collections,itertools

def main():
 assert not(E/'configs/frozen.json').exists();source=read(E/'configs/source_references.json')
 for p,h in source['sha256'].items():assert sha(p)==h,p
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
 supports=[]
 for split,folder in [('train','train'),('validation','validation'),('diagnostic_D_seen','evaluator_D')]:
  rr=read(E/f'data/{folder}/objects.json')
  for color,held in itertools.product(COLORS,COLORS):
   ss=[r for r in rr if r['object_color']==color and r['held_color']==held];p=[r for r in ss if r['In']];n=[r for r in ss if not r['In']];supports.append({'split':split,'object_color':color,'held_color':held,'positive_N':len(p),'negative_N':len(n),'positive_roots':sorted({r['root_id']for r in p}),'negative_roots':sorted({r['root_id']for r in n})})
   if p:assert len(n)>0,'No matched identity negatives'
 save(E/'results/matched_identity_support.json',supports)
 cfg=read(E/'configs/protocol.json');cfg['training']='CPU fullbatch AdamW lr0.001 wd0.001,500updates; BCEWithLogits weighted so each root and each within-root class contributes equally; perinput standardization fromtrainonly,std floor0.01; shared3objectclassifier,no colour/held/sequence/history/physics inputs.';cfg['matched_identity_control']='For every positive(objectcolor,heldcolor)combination, same split includes negative examples with identical objectcolor+heldcolor; metadata neverinput toprobe.';save(E/'configs/protocol.json',cfg)
 sources=dict(source['sha256'])
 for p in E.rglob('*'):
  if p.is_file()and '__pycache__'not in p.parts:sources[str(p)]=sha(p)
 save(E/'configs/frozen.json',{'utc':now(),'probe_training_started':False,'fits_allowed':4,'no_hyperparameter_search':True,'sha256':sources,'prior_manifests':source['prior_manifests']});print('PROBE_FROZEN',len(sources),'refs;4fits,500updates each;AB6/6,Dseen6',flush=True)
if __name__=='__main__':main()
