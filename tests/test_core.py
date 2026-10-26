import unittest
import numpy as np
import pandas as pd
from src.hdfs_study.core import merge_data, split_indices, metrics, choose_thresholds, NormalRule

class CoreTests(unittest.TestCase):
    def fixtures(self):
        x = pd.DataFrame({'BlockId':['b','a'], 'E1':[3,1], 'Label':['bad','bad'], 'Type':['leak','leak']})
        labels = pd.DataFrame({'BlockId':['a','b'], 'Label':['Anomaly','Normal']})
        return x, labels
    def test_keyed_join_not_row_order(self):
        x, labels = self.fixtures(); joined, _ = merge_data(x, labels)
        self.assertEqual(joined.truth.tolist(), ['Anomaly','Normal'])
        self.assertEqual(joined.E1.tolist(), [1,3])
    def test_no_label_or_id_features(self):
        self.assertEqual(merge_data(*self.fixtures())[1], ['E1'])
    def test_duplicate_missing_and_unknown_labels_fail(self):
        x, y = self.fixtures()
        with self.assertRaises(ValueError): merge_data(pd.concat([x,x]), y)
        with self.assertRaises(ValueError): merge_data(x,y.iloc[:1])
        y.loc[0,'Label']='unknown'
        with self.assertRaises(ValueError): merge_data(x,y)
    def test_invalid_counts_fail(self):
        x,y=self.fixtures(); x.loc[0,'E1']=-1
        with self.assertRaises(ValueError):merge_data(x,y)
    def test_disjoint_complete_deterministic_split(self):
        y=np.r_[np.zeros(900),np.ones(100)]
        a,b=split_indices(y),split_indices(y)
        self.assertEqual([len(v) for v in a.values()],[600,200,200])
        self.assertEqual(len(set(np.concatenate(list(a.values())))),1000)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_metrics_known_confusion(self):
        m=metrics([1,1,0,0],[.9,.1,.8,.2],.5)
        for k in ('tp','fp','fn','tn'):self.assertEqual(m[k],1)
        for k in ('precision','recall','f1','fpr'):self.assertEqual(m[k],.5)
    def test_undefined_precision(self):
        m=metrics([0,1],[0.,0.],1.)
        self.assertTrue(np.isnan(m['precision']));self.assertEqual(m['f1'],0.)
    def test_ties_and_fpr_constraint(self):
        y=np.r_[np.ones(2),np.zeros(100)]; s=np.ones(102)
        t=choose_thresholds(y,s)
        self.assertGreater(t['fpr01'],1)
        self.assertEqual(metrics(y,s,t['fpr01'])['fpr'],0)
        self.assertEqual(t['f1'],1.)
    def test_rule_fits_normals_only(self):
        r=NormalRule().fit(np.array([[1.],[3.],[999.]]),[0,0,1])
        np.testing.assert_equal(r.mean_,[2.]);np.testing.assert_equal(r.score(np.array([[4.]])),[2.])
    def test_int8_positive_support_does_not_overflow(self):
        y=np.r_[np.ones(256,dtype=np.int8),np.zeros(256,dtype=np.int8)]
        m=metrics(y,y.astype(float),.5)
        self.assertEqual(m['average_precision'],1.);self.assertEqual(m['pr_auc'],1.)
    def test_ap_is_not_trapezoidal_area(self):
        m=metrics([0,1],[0.,0.],0.)
        self.assertEqual(m['average_precision'],.5);self.assertEqual(m['pr_auc'],.75)

if __name__=='__main__':unittest.main()
