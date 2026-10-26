"""Validate all CSV joins and trace counts; create deterministic local split artifacts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,json,re,zipfile
import numpy as np
import pandas as pd
from src.hdfs_study.core import merge_data,split_indices,SEED

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    raw=ROOT/'data/raw/preprocessed'; out=ROOT/'data/processed';out.mkdir(exist_ok=True)
    result=ROOT/'results/data';result.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for f in manifest['files']:
        assert sha(ROOT/f['path'])==f['sha256'],f['path']
    matrix=pd.read_csv(raw/'Event_occurrence_matrix.csv');labels=pd.read_csv(raw/'anomaly_label.csv')
    joined,features=merge_data(matrix,labels)
    assert joined.Label.map({'Success':'Normal','Fail':'Anomaly'}).eq(joined.truth).all()
    ids=joined.BlockId.to_numpy(dtype=str);x=joined[features].to_numpy(dtype=np.int32)
    y=joined.truth.eq('Anomaly').to_numpy(dtype=np.int8)
    # Trace CSV has event tokens but no stable raw-line identity. Verify all counts by keyed lookup.
    lookup={b:i for i,b in enumerate(ids)}; seen=set();mismatch=0
    for chunk in pd.read_csv(raw/'Event_traces.csv',usecols=['BlockId','Label','Features'],chunksize=50000):
        for b,label,trace in chunk.itertuples(index=False,name=None):
            if b in seen or b not in lookup:raise ValueError('Trace ID mismatch/duplicate')
            seen.add(b);i=lookup[b]
            if label != ('Fail' if y[i] else 'Success'):raise ValueError('Trace label mismatch')
            if not re.fullmatch(r'\[E[1-9][0-9]*(?:,E[1-9][0-9]*)*\]',trace):raise ValueError('Invalid trace syntax')
            tokens=np.fromiter((int(t[1:])-1 for t in trace[1:-1].split(',')),dtype=int)
            if tokens.max()>=len(features):raise ValueError('Unknown trace event')
            mismatch += not np.array_equal(np.bincount(tokens,minlength=len(features)),x[i])
    assert len(seen)==len(ids) and mismatch==0,'Trace/matrix count mismatch'
    # Inspect NPZ .npy headers without deserializing object data; this file is NOT used for modeling.
    headers={}
    with zipfile.ZipFile(raw/'HDFS.npz') as z:
        for name in z.namelist():
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None: raise ValueError('Unsupported NPY header version')
                shape,fortran,dtype=reader(f)
                headers[name]={'shape':list(shape),'dtype':str(dtype),'contains_objects':dtype.hasobject}
    splits=split_indices(y)
    assignment=np.empty(len(y),dtype='U10')
    rows=[]
    for name,ix in splits.items():
        assignment[ix]=name
        rows.append(dict(split=name,blocks=len(ix),normal=int((y[ix]==0).sum()),anomaly=int(y[ix].sum()),anomaly_fraction=float(y[ix].mean())))
    pd.DataFrame(rows).to_csv(result/'sample_counts.csv',index=False)
    pd.DataFrame({'BlockId':ids,'split':assignment}).to_csv(out/'split.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    np.savez_compressed(out/'dataset.npz',ids=ids,x=x,y=y,features=np.array(features),**splits)
    duplicates=[]
    groups={}
    for representation,a in [('frequency',x),('presence',(x>0).astype(np.int32))]:
        unique,codes,counts=np.unique(a,axis=0,return_inverse=True,return_counts=True)
        groups[representation]=codes
        labels_min=np.full(len(unique),2);labels_max=np.full(len(unique),-1)
        np.minimum.at(labels_min,codes,y);np.maximum.at(labels_max,codes,y)
        train_codes=np.unique(codes[splits['train']])
        for name,ix in splits.items():
            other_ix=np.flatnonzero(assignment!=name)
            shared=np.isin(codes[ix],np.unique(codes[other_ix]))
            duplicates.append(dict(representation=representation,split=name,blocks=len(ix),unique_vectors=int(len(np.unique(codes[ix]))),blocks_with_vector_in_another_split=int(shared.sum()),fraction_shared=float(shared.mean()),blocks_with_vector_in_train=int(np.isin(codes[ix],train_codes).sum()),global_unique_vectors=len(unique),global_conflicting_label_vectors=int((labels_min!=labels_max).sum()),global_duplicate_rows=int(len(y)-len(unique))))
    pd.DataFrame(duplicates).to_csv(result/'duplicate_vectors.csv',index=False)
    np.savez_compressed(out/'feature_groups.npz',**groups)
    quality={'rows':len(y),'normal':int((y==0).sum()),'anomaly':int(y.sum()),'anomaly_fraction':float(y.mean()),'features':features,'missing_or_duplicate_ids':0,'unmatched_ids':0,'invalid_counts':0,'trace_rows_verified':len(seen),'trace_count_mismatches':int(mismatch),'matrix_label_mismatches':0,'npz_headers':headers,'npz_used':False,'shared_raw_line_isolation':'UNVERIFIED: trace CSV has event tokens, not original line IDs','upstream_vocabulary':'Official full-data preprocessing; not training-fitted by this study','seed':SEED,'split_sha256':sha(out/'split.csv.gz'),'constant_features':[features[i] for i in range(len(features)) if np.unique(x[:,i]).size==1]}
    (result/'data_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    print(json.dumps(quality,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(duplicates).to_string(index=False))
if __name__=='__main__':main()
