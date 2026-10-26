"""Acquire the fixed official HDFS_v1 archive and extract only inspected preprocessed members."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,json,shutil,stat,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
URL='https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1'
EXPECTED_BYTES=186645559
EXPECTED_MD5='76a24b4d9a6164d543fb275f89773260'
NAMES={'anomaly_label.csv','Event_occurrence_matrix.csv','Event_traces.csv','HDFS.npz','HDFS_templates.csv'}
def digest(p,algorithm='sha256'):
 h=hashlib.new(algorithm)
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def main():
 raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(raw).free>3_000_000_000,'At least 3 GB free is required'
 license_url='https://raw.githubusercontent.com/logpai/loghub/master/LICENSE'
 license_path=raw/'LOGHUB_LICENSE.txt'
 if not license_path.exists():
  with urllib.request.urlopen(license_url,timeout=30) as r:text=r.read(100_000).decode()
  if 'research or academic work' not in text or 'included in all copies' not in text:raise ValueError('Review changed licence')
  license_path.write_text(text);license_path.chmod(0o444)
 archive=raw/'HDFS_v1.zip';acquired=False
 if not archive.exists():
  temp=archive.with_suffix('.part')
  with urllib.request.urlopen(URL,timeout=60) as response,temp.open('xb') as f:
   total=0;last=0
   while b:=response.read(1024*1024):
    total+=len(b)
    if total>250_000_000:raise ValueError('Download exceeds budget')
    f.write(b)
    if total-last>=20_000_000:print('Downloaded bytes:',total,flush=True);last=total
  if temp.stat().st_size!=EXPECTED_BYTES or digest(temp,'md5')!=EXPECTED_MD5:raise ValueError('Publisher size/MD5 mismatch')
  temp.rename(archive);archive.chmod(0o444);acquired=True
 assert archive.stat().st_size==EXPECTED_BYTES and digest(archive,'md5')==EXPECTED_MD5
 with zipfile.ZipFile(archive) as z:
  infos=z.infolist();inventory=[{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size} for i in infos]
  (ROOT/'data/archive_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
  wanted=[i for i in infos if PurePosixPath(i.filename).name in NAMES and '__MACOSX' not in PurePosixPath(i.filename).parts]
  if not wanted:raise ValueError('No expected preprocessed members; inspect inventory')
  if sum(i.file_size for i in wanted)>1_000_000_000:raise ValueError('Extraction budget exceeded')
  seen=set();files=[]
  for i in wanted:
   parts=PurePosixPath(i.filename);name=parts.name;mode=i.external_attr>>16
   if parts.is_absolute() or '..' in parts.parts or '\\' in i.filename or stat.S_ISLNK(mode) or name in seen:raise ValueError('Unsafe/duplicate member')
   seen.add(name);dest=raw/'preprocessed'/name;dest.parent.mkdir(exist_ok=True)
   if not dest.exists():
    with z.open(i) as src,dest.open('xb') as out:shutil.copyfileobj(src,out,1024*1024)
    dest.chmod(0o444)
   assert dest.stat().st_size==i.file_size
   files.append({'archive_member':i.filename,'path':str(dest.relative_to(ROOT)),'bytes':i.file_size,'sha256':digest(dest)})
 manifest={'dataset':'HDFS_v1','zenodo_record':'8196385','record_version':'v8 (not HDFS version)','url':URL,'archive_bytes':archive.stat().st_size,'archive_md5':digest(archive,'md5'),'archive_sha256':digest(archive),'files':files,'raw_log_extracted':False,'licence_url':license_url,'checked_at':datetime.now(timezone.utc).isoformat()}
 if not (ROOT/'data/manifest.json').exists() or acquired:(ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 else:
  previous=json.loads((ROOT/'data/manifest.json').read_text());assert previous['archive_sha256']==manifest['archive_sha256'] and previous['files']==files
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
