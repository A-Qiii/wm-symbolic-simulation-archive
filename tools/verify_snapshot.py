"""Read-only archive checks; no model, simulator or network calls."""
from pathlib import Path
import sys,json,hashlib,re,ast,argparse
S=Path(__file__).resolve().parent.parent
PATTERNS={
 'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
 'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{30,255})\b',
 'openai_key':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b',
 'aws_access_id':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'slack_token':r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
 'credential_in_url':r'https?://[^/\s:@]+:[^/@\s]+@',
 'literal_credential':r'''(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|client[_-]?secret|secret[_-]?key)["']?\s*[:=]\s*["']([^"'\n]{12,})["']''',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for a in iter(lambda:f.read(4194304),b''):h.update(a)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check-original',action='store_true');args=ap.parse_args();errors=[];alerts=[];nc=np=nj=nt=0
 manifest=json.loads((S/'SNAPSHOT_MANIFEST.json').read_text())
 for a in manifest['files']:
  p=S/a['path']
  if not p.is_file()or p.is_symlink()or p.stat().st_size!=a['bytes']or sha(p)!=a['sha256']:errors.append({'path':a['path'],'check':'snapshot_integrity'})
  nc+=1
 for p in S.rglob('*'):
  if not p.is_file()or'.git'in p.relative_to(S).parts:continue
  rel=str(p.relative_to(S))
  if p.suffix in {'.png','.npz'}:continue
  try:text=p.read_text(encoding='utf-8')
  except UnicodeError:errors.append({'path':rel,'check':'unexpected_binary'});continue
  nt+=1
  for kind,pattern in PATTERNS.items():
   for m in re.finditer(pattern,text):alerts.append({'path':rel,'line':text.count('\n',0,m.start())+1,'rule':kind})
  if p.suffix=='.py':
   try:ast.parse(text,filename=rel);np+=1
   except SyntaxError as e:errors.append({'path':rel,'check':'python_syntax','line':e.lineno,'message':e.msg})
  if p.suffix=='.json':
   try:json.loads(text);nj+=1
   except json.JSONDecodeError as e:errors.append({'path':rel,'check':'json_parse','line':e.lineno})
 if args.check_original:
  for a in json.loads((S/'provenance/copied_sources.json').read_text()):
   p=Path(a['source'])
   if not p.is_file()or sha(p)!=a['sha256']:errors.append({'path':a['path'],'check':'original_hash_unchanged'})
 r={'passed':not errors and not alerts,'manifest_files_checked':nc,'text_files_secret_scanned':nt,'python_syntax_passed':np,'json_parse_passed':nj,'original_sources_checked':args.check_original,'high_confidence_credential_alerts':alerts,'errors':errors,'scope':'Checksums, syntax,JSON and high-confidence credential patterns only. No training,collection,GPU/model inference,or full reproducibility test.'}
 print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['passed']else 1
if __name__=='__main__':raise SystemExit(main())
