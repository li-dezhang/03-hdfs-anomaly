"""Compare a reproduced final run and verify the checked-in report tables and local links."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compare',required=True);args=parser.parse_args()
    original=ROOT/'results/final-20261021';new=ROOT/'results'/args.compare
    checks={}
    for name in ['metrics.csv','validation_metrics.csv','sample_counts.csv']:
        a,b=pd.read_csv(original/name),pd.read_csv(new/name)
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-12,atol=1e-14)
        checks[name]='PASS'
    metrics=pd.read_csv(original/'metrics.csv');body=(ROOT/'report/RESEARCH_REPORT.md').read_text();readme=(ROOT/'README.md').read_text()
    for r in metrics[metrics.operating_point=='f1'].itertuples():
        expected='| '+r.representation+' | '+r.method+' | '+' | '.join(f'{getattr(r,c):.6f}' for c in ['precision','recall','f1','pr_auc','average_precision','fpr'])+' |'
        assert expected in body,'Report metric table differs from CSV'
        expected='| '+r.representation+' | '+r.method+' | '+' | '.join(f'{getattr(r,c):.4f}' for c in ['precision','recall','f1','pr_auc'])+' |'
        assert expected in readme,'README metric table differs from CSV'
    validation=pd.read_csv(original/'validation_metrics.csv')
    for r in metrics[metrics.operating_point=='fpr01'].itertuples():
        v=validation[(validation.representation==r.representation)&(validation.method==r.method)&(validation.operating_point=='fpr01')].iloc[0]
        expected='| '+' | '.join([r.representation,r.method,f'{v.fpr:.6f}',f'{r.fpr:.6f}',f'{r.recall:.6f}',str(r.tp),str(r.fp),str(r.fn)])+' |'
        assert expected in body,'Report constrained table differs from CSV'
    for r in pd.read_csv(original/'sample_counts.csv').itertuples():
        expected='| '+' | '.join([r.split,f'{r.blocks:,}',f'{r.normal:,}',f'{r.anomaly:,}',f'{100*r.anomaly_fraction:.4f}'])+' |'
        assert expected in body,'Report sample table differs from CSV'
    links=0
    for p in [ROOT/'README.md',ROOT/'report/RESEARCH_REPORT.md',ROOT/'RESEARCH_BRIEF.md',ROOT/'data/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http://','https://','#')):continue
            path=target.split('#')[0]
            assert (p.parent/path).exists(),f'Broken link: {p}: {target}'
            links+=1
    score_deltas={}
    for rep in ['frequency','presence']:
        for method in ['rule','logistic','forest']:
            for split in ['validation','test']:
                name=f'{rep}__{method}_{split}.npz'
                a=ROOT/'data/processed/final-20261021'/name;b=ROOT/'data/processed'/args.compare/name
                if a.exists() and b.exists():
                    x,y=np.load(a,allow_pickle=False),np.load(b,allow_pickle=False)
                    np.testing.assert_array_equal(x['index'],y['index']);np.testing.assert_array_equal(x['y'],y['y'])
                    np.testing.assert_allclose(x['scores'],y['scores'],rtol=1e-12,atol=1e-14)
                    score_deltas[name]=float(np.max(np.abs(x['scores']-y['scores'])))
    record={'status':'PASS','compared_run':args.compare,'csv_checks':checks,'report_and_readme_metric_tables_match':True,'local_links_checked':links,'score_max_abs_deltas':score_deltas,'scope':'same-platform deterministic rerun; no new test-driven choices','test_metric_rows':len(metrics)}
    # Write alongside the rerun, not over the original verification evidence.
    (new/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
