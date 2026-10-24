from common import *
def main():
 manifest=read(E/'configs/delivery_manifest.json')
 for path,item in manifest['files'].items():
  assert Path(path).stat().st_size==item['bytes'],path
  assert sha(path)==item['sha256'],path
 for name in ['frozen.json','evaluation_frozen.json','models_frozen.json','probes_frozen.json']:
  for path,h in read(E/'configs'/name)['sha256'].items():assert sha(path)==h,path
 print(json.dumps({'passed':True,'sealed_files':len(manifest['files']),'bytes':sum(v['bytes']for v in manifest['files'].values()),'manifest_sha256':sha(E/'configs/delivery_manifest.json'),'read_only_verification':True},ensure_ascii=False))
if __name__=='__main__':main()
