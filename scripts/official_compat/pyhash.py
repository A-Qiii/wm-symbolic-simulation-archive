"""Diagnostic-only subset of pyhash 0.9.3: fnv1_32 on ASCII strings/bytes.

Source: official PyPI pyhash-0.9.3.tar.gz, SHA256
cff5c81d613163fc59d623d4546d9be55b46ecd0e573b59057b1bb112a497763.
Hash.h converts Python str to native UTF-16 without BOM; seed defaults to 0.
FNV1.h calls fnv_32_buf from public-domain src/fnv/hash_32.c.
This is explicitly a compatibility implementation, not the installed package.
Only put its containing directory on this isolated diagnostic's sys.path.
"""
import sys
class fnv1_32:
 def __init__(self,seed=0):
  if sys.byteorder!='little':raise RuntimeError('Verified only for this little-endian local host')
  self.seed=seed
 def __call__(self,*args,seed=None):
  value=self.seed if seed is None else seed
  if not isinstance(value,int) or not 0<=value<=0xffffffff:raise ValueError('Expected uint32 seed')
  for data in args:
   if isinstance(data,str):
    if not data.isascii():raise ValueError('Only CALVIN ASCII initial-condition strings are validated')
    data=data.encode('utf-16-le')
   if not isinstance(data,bytes):raise TypeError('Only bytes/ASCII str supported by diagnostic subset')
   for byte in data:value=((value*0x01000193)&0xffffffff)^byte
  return value
