"""Fixed three-method, two-representation benchmark; test only after thresholds freeze."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,platform,time,warnings
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.hdfs_study.core import SEED,split_indices,metrics,choose_thresholds,NormalRule

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','final'],required=True);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    result=ROOT/'results'/args.run_id;result.mkdir(parents=True,exist_ok=False)
    local=ROOT/'data/processed'/args.run_id;local.mkdir(exist_ok=False)
    d=np.load(ROOT/'data/processed/dataset.npz',allow_pickle=False)
    x,y,ids=d['x'],d['y'],d['ids'];splits={k:d[k] for k in ['train','validation','test']}
    if args.mode=='pilot':
        pool=splits['train'];picked,_=train_test_split(pool,train_size=30000,stratify=y[pool],random_state=SEED)
        picked=np.sort(picked);splits={k:picked[v] for k,v in split_indices(y[picked]).items()}
        assert not set(picked)&set(d['test']) and not set(picked)&set(d['validation'])
    splitrows=[{'split':k,'blocks':len(v),'normal':int((y[v]==0).sum()),'anomaly':int(y[v].sum()),'anomaly_fraction':float(y[v].mean())} for k,v in splits.items()]
    pd.DataFrame(splitrows).to_csv(result/'sample_counts.csv',index=False)
    models={};thresholds={};valrows=[];timings=[];params={}
    started=datetime.now(timezone.utc).isoformat()
    warnings.filterwarnings('error',category=ConvergenceWarning)
    with threadpool_limits(limits=2):
        for rep in ['frequency','presence']:
            a=x.astype(np.float64) if rep=='frequency' else (x>0).astype(np.float64)
            train,valid=splits['train'],splits['validation']
            for method in ['rule','logistic','forest']:
                key=f'{rep}__{method}';start=time.perf_counter()
                if method=='rule':model=NormalRule()
                elif method=='logistic':
                    steps=[]
                    if rep=='frequency':steps.append(('log1p',FunctionTransformer(np.log1p)))
                    steps.extend([('scale',StandardScaler()),('model',LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=SEED))])
                    model=Pipeline(steps)
                else:model=RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=5,max_features='sqrt',class_weight='balanced_subsample',n_jobs=2,random_state=SEED)
                model.fit(a[train],y[train]);fit_seconds=time.perf_counter()-start
                scores=model.score(a[valid]) if method=='rule' else model.predict_proba(a[valid])[:,1]
                thresholds[key]=choose_thresholds(y[valid],scores)
                for op,t in thresholds[key].items():valrows.append(dict(representation=rep,method=method,operating_point=op,threshold=t,**metrics(y[valid],scores,t)))
                np.savez_compressed(local/f'{key}_validation.npz',index=valid,y=y[valid],scores=scores)
                models[key]=model;params[key]=str(model.get_params()) if hasattr(model,'get_params') else 'normal mean/std, ddof=0, scale_floor=1'
                timings.append(dict(representation=rep,method=method,fit_seconds=fit_seconds))
                print(f'FIT {key}: {fit_seconds:.3f}s; validation thresholds {thresholds[key]}',flush=True)
    # Persist complete threshold set before accessing test labels or computing test metrics.
    (result/'thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    pd.DataFrame(valrows).to_csv(result/'validation_metrics.csv',index=False)
    frozen_at=datetime.now(timezone.utc).isoformat();rows=[]
    with threadpool_limits(limits=2):
        for rep in ['frequency','presence']:
            a=x.astype(np.float64) if rep=='frequency' else (x>0).astype(np.float64)
            test=splits['test']
            for method in ['rule','logistic','forest']:
                key=f'{rep}__{method}';model=models[key];start=time.perf_counter()
                scores=model.score(a[test]) if method=='rule' else model.predict_proba(a[test])[:,1]
                assert len(scores)==len(test)
                np.savez_compressed(local/f'{key}_test.npz',index=test,y=y[test],scores=scores)
                for op,t in thresholds[key].items():rows.append(dict(representation=rep,method=method,operating_point=op,threshold=t,**metrics(y[test],scores,t)))
                timings[[f'{r["representation"]}__{r["method"]}' for r in timings].index(key)]['test_score_seconds']=time.perf_counter()-start
    pd.DataFrame(rows).to_csv(result/'metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(result/'timings.csv',index=False)
    provenance={'mode':args.mode,'run_id':args.run_id,'started_utc':started,'thresholds_frozen_utc':frozen_at,'completed_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,'seed':SEED,'parameters':params,'protocol_sha256':hashlib.sha256((ROOT/'docs/EXPERIMENT_PROTOCOL.md').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/hdfs_study/core.py',ROOT/'scripts/prepare.py']}}
    (result/'run.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
