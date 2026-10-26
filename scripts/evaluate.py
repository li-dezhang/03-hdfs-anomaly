"""Rescore saved predictions, verify all experiments, create diagnostics and figures."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from src.hdfs_study.core import metrics,choose_thresholds

METHODS=['rule','logistic','forest'];REPS=['frequency','presence']
LABELS={'rule':'Normal-profile rule','logistic':'Logistic regression','forest':'Random forest'}
COLORS={'rule':'#7B6D8D','logistic':'#1976A3','forest':'#2D8B65'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
    result=ROOT/'results'/args.run_id;local=ROOT/'data/processed'/args.run_id
    figs=ROOT/'figures'/args.run_id;figs.mkdir(parents=True,exist_ok=True)
    thresholds=json.loads((result/'thresholds.json').read_text());rows=[];vrows=[];curves=[];subsets=[];errors=[]
    data=np.load(ROOT/'data/processed/dataset.npz',allow_pickle=False)
    groups=np.load(ROOT/'data/processed/feature_groups.npz',allow_pickle=False)
    final=json.loads((result/'run.json').read_text())['mode']=='final'
    for rep in REPS:
        for method in METHODS:
            key=f'{rep}__{method}'
            val=np.load(local/f'{key}_validation.npz',allow_pickle=False)
            assert choose_thresholds(val['y'],val['scores'])==thresholds[key]
            test=np.load(local/f'{key}_test.npz',allow_pickle=False)
            ix,y,s=test['index'],test['y'],test['scores']
            assert np.array_equal(y,data['y'][ix]) and len(ix)==len(np.unique(ix))
            if final:assert np.array_equal(ix,data['test'])
            for op,t in thresholds[key].items():
                rows.append(dict(representation=rep,method=method,operating_point=op,threshold=t,**metrics(y,s,t)))
                vrows.append(dict(representation=rep,method=method,operating_point=op,threshold=t,**metrics(val['y'],val['scores'],t)))
            precision,recall,ts=precision_recall_curve(y,s)
            for pr,re,th in zip(precision,recall,np.r_[ts,np.nan]):curves.append(dict(representation=rep,method=method,precision=pr,recall=re,threshold=th))
            if final:
                codes=groups[rep];seen=np.isin(codes[ix],np.unique(codes[data['train']]))
                for label,mask in [('seen_in_train',seen),('unseen_in_train',~seen)]:
                    if mask.any():subsets.append(dict(representation=rep,method=method,subset=label,**metrics(y[mask],s[mask],thresholds[key]['f1'])))
                pred=s>=thresholds[key]['f1']
                for kind,mask in [('FP',(y==0)&pred),('FN',(y==1)&~pred)]:
                    vals,counts=np.unique(codes[ix[mask]],return_counts=True)
                    for g,n in sorted(zip(vals,counts),key=lambda a:(-a[1],a[0]))[:5]:
                        selected=np.flatnonzero(mask&(codes[ix]==g));example=ix[selected[0]]
                        trainmask=codes[data['train']]==g
                        features=data['x'][example] if rep=='frequency' else (data['x'][example]>0).astype(int)
                        errors.append(dict(representation=rep,method=method,error=kind,pattern_id=int(g),test_errors=int(n),score=float(s[selected[0]]),threshold=thresholds[key]['f1'],train_same_pattern_normal=int(((data['y'][data['train']]==0)&trainmask).sum()),train_same_pattern_anomaly=int(((data['y'][data['train']]==1)&trainmask).sum()),nonzero_features=';'.join(f'{e}={v}' for e,v in zip(data['features'],features) if v)))
    got=pd.DataFrame(rows);vgot=pd.DataFrame(vrows)
    pd.testing.assert_frame_equal(got,pd.read_csv(result/'metrics.csv'),check_exact=False,rtol=1e-12,atol=1e-14)
    pd.testing.assert_frame_equal(vgot,pd.read_csv(result/'validation_metrics.csv'),check_exact=False,rtol=1e-12,atol=1e-14)
    got.to_csv(result/'rescored_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(result/'pr_curves.csv',index=False)
    if final:
        pd.DataFrame(subsets).to_csv(result/'seen_unseen_metrics.csv',index=False)
        pd.DataFrame(errors).to_csv(result/'error_patterns.csv',index=False)
    # Read the exported tables back: these exact public CSVs are the figure inputs.
    m=pd.read_csv(result/'metrics.csv');pr=pd.read_csv(result/'pr_curves.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax,rep in zip(axes,REPS):
        sub=m[(m.representation==rep)&(m.operating_point=='f1')].set_index('method').loc[METHODS]
        pos=np.arange(3);width=.24
        for j,metric in enumerate(['precision','recall','f1']):
            bars=ax.bar(pos+(j-1)*width,sub[metric],width,label=metric.title())
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
        ax.set(xticks=pos,xticklabels=['Rule','Logistic','Forest'],ylim=(0,1.12),ylabel='Test metric',title=rep.title()+' features');ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Test performance at validation-selected F1 thresholds');fig.savefig(figs/'performance.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,rep in zip(axes,REPS):
        for method in METHODS:
            q=pr[(pr.representation==rep)&(pr.method==method)]
            row=m[(m.representation==rep)&(m.method==method)].iloc[0]
            ax.plot(q.recall,q.precision,label=f'{LABELS[method]} (area={row.pr_auc:.3f})',color=COLORS[method],linewidth=1.8)
        prevalence=row['tp']+row['fn'];prevalence/=row['n']
        ax.axhline(prevalence,color='gray',linestyle=':',label=f'Prevalence {prevalence:.3f}')
        ax.set(xlim=(0,1.01),ylim=(0,1.03),xlabel='Recall',ylabel='Precision',title=rep.title());ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Test precision–recall curves (trapezoidal area)');fig.savefig(figs/'precision_recall.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for i,rep in enumerate(REPS):
        for j,method in enumerate(METHODS):
            r=m[(m.representation==rep)&(m.method==method)&(m.operating_point=='f1')].iloc[0]
            a=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int);norm=a/a.sum(axis=1,keepdims=True)
            ax=axes[i,j];ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
            for k in range(2):
                for l in range(2):ax.text(l,k,f'{a[k,l]:,}\n({norm[k,l]:.2%})',ha='center',va='center',color='white' if norm[k,l]>.5 else 'black')
            ax.set(xticks=[0,1],xticklabels=['Normal','Anomaly'],yticks=[0,1],yticklabels=['Normal','Anomaly'],xlabel='Predicted',ylabel='Actual',title=f'{rep} / {method}')
    fig.suptitle('Test confusion matrices at validation-F1 thresholds; row-normalized color');fig.savefig(figs/'confusion_matrices.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');v=pd.read_csv(result/'validation_metrics.csv')
    keys=[(rep,method) for rep in REPS for method in METHODS];names=[f'{r}\n{m}' for r,m in keys];pos=np.arange(6)
    for ax,metric,title in zip(axes,['fpr','recall'],['Actual false-positive rate','Anomaly recall']):
        for j,(table,label) in enumerate([(v,'Validation'),(m,'Test')]):
            values=[table[(table.representation==rep)&(table.method==method)&(table.operating_point=='fpr01')].iloc[0][metric] for rep,method in keys]
            ax.bar(pos+(j-.5)*.35,values,.35,label=label)
        if metric=='fpr':ax.axhline(.01,color='red',linestyle='--',label='Validation target: 1%')
        ax.set(xticks=pos,xticklabels=names,title=title,ylabel='Fraction');ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
    fig.suptitle('Operating points selected to maximize validation recall with FPR ≤ 1%');fig.savefig(figs/'fpr_operating_point.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,3.7),layout='constrained');ax.set(xlim=(0,11),ylim=(0,4));ax.axis('off')
    boxes=[(1.3,3,'Official ZIP\nID-keyed CSV join'),(4,3,'Validate counts\n575,061 blocks' if final else 'Pilot from final\ntraining pool only'),(7,3,'60 / 20 / 20\nstratified Block split'),(9.8,3,'Train-only fit\n3 methods × 2 features'),(9.8,1,'Validation thresholds\nF1 and FPR ≤ 1%'),(6.8,1,'Freeze choices\nScore test once'),(3.8,1,'Rescore predictions\nCSV → figures'),(1.2,1,'Report limitations\nNo deployment claim')]
    for x0,y0,label in boxes:ax.text(x0,y0,label,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.5',facecolor='#E9F1F7',edgecolor='#3C6685'))
    for (x1,y1,_),(x2,y2,_) in zip(boxes,boxes[1:]):
        dx,dy=x2-x1,y2-y1;ax.annotate('',xy=(x2-np.sign(dx)*1.0,y2-np.sign(dy)*.4),xytext=(x1+np.sign(dx)*1.0,y1+np.sign(dy)*.4),arrowprops=dict(arrowstyle='->',color='#3C6685'))
    fig.savefig(figs/'workflow.png');plt.close(fig)
    verification={'status':'PASS','test_metric_rows':len(got),'validation_metric_rows':len(vgot),'thresholds_reselected_from_validation_identically':True,'metric_comparison_rtol':1e-12,'all_six_configurations_present':len(thresholds)==6,'figures_from_exported_csv':True,'inputs_sha256':{n:hashlib.sha256((result/n).read_bytes()).hexdigest() for n in ['metrics.csv','pr_curves.csv','validation_metrics.csv']}}
    (result/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2));print('Figures:',figs)
if __name__=='__main__':main()
