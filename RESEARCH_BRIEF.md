# P3 — HDFS Log Anomaly Detection

Current scope (2026-10-21): empirical research using official HDFS_v1 preprocessed block features. The user explicitly authorized public academic acquisition, programming, experiments and reporting. The previous Gate A was never approved; its historical design is preserved in [docs/history/RESEARCH_BRIEF_PHASE0.md](docs/history/RESEARCH_BRIEF_PHASE0.md). It does not govern this lightweight workflow. No publication approval is implied.

**Title:** Detecting Anomalous Events in Distributed System Logs Using Machine Learning: A Comparative Evaluation on HDFS_v1.

**Main question:** To what extent do machine learning models improve block-level anomaly detection compared with a simple statistical rule?

- RQ1: How do the rule, Logistic Regression and Random Forest differ in precision, recall and F1?
- RQ2: At a validation FPR target of at most 1%, do learned models detect more anomalous blocks?
- RQ3: How do event frequency and event presence representations change performance?

**Data:** official Loghub / Zenodo record 8196385, HDFS_v1.zip; exact URLs, sizes and hashes in [data/manifest.json](data/manifest.json). The record's v8 designation is not HDFS_v8. Only four preprocessed files are extracted. The modeling inputs are Event_occurrence_matrix.csv and anomaly_label.csv, joined one-to-one by BlockId. Event_traces.csv independently checks event counts. HDFS.npz headers are inspected without loading pickled objects; NPZ is not an input.

**Unit and reference:** one Block ID, 29 count features, Normal/Anomaly labels supplied by the corpus. These are curated benchmark labels, not independent network-attack truth. Matrix Label and Type are excluded from predictors. All 575,061 IDs are matched; no exclusions were necessary.

**Experiment:** [fixed protocol](docs/EXPERIMENT_PROTOCOL.md), seed 20261021, stratified 60/20/20 Block split. Pilot uses 30,000 blocks drawn only from final training. Three predeclared methods × two representations, same split; training-only model fitting; validation-only F1 and constrained-FPR thresholds frozen before final evaluation. No hyperparameter search. See [report](report/RESEARCH_REPORT.md) for actual results.

**Metrics:** anomaly is positive. Precision denominator TP+FP; recall TP+FN; F1 denominator 2TP+FP+FN; FPR FP+TN. Undefined precision is blank. Trapezoidal PR-AUC and average precision are separately reported. Scores cover every test block. No causal or independent-sample inference is claimed.

**Risks:** identical feature patterns cross splits extensively; shared original log lines are not isolated; the publisher's upstream vocabulary is reused, not fitted on training logs; labels may closely reflect count patterns. Thus this tests familiar-pattern classification within one environment, not new-failure, time-forward, or cross-system generalization. Counts in older publications differ from this archived version.

**Deliverables:** executable Python, keyed data validation, unit tests, preserved run logs, genuine CSV metrics, plots, English report and reproduction instructions. Raw data, local predictions and environments are excluded from Git. Researcher review of interpretation and portfolio authorship remains necessary; remote publication is deferred until the user requests it. No P1/P2/P4/P5 work is included.
