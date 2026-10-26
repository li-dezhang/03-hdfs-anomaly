# Fixed experiment protocol — 2026-10-21

Written before pilot/model fitting. This replaces the old raw-log parsing design at the user's request. Historical design is retained in docs/history. No old approval is retrospectively granted.

- Unit: unique BlockId in the official matrix; exact one-to-one keyed label join. Only E1…E29 counts are predictors. Label, Type, ID, time and latency are excluded.
- Official upstream event vocabulary is reused; it was not fitted within our training split. Raw-line sharing across blocks is unverified because the selected preprocessed files have no raw-line identifiers. This is a limitation, not a passed isolation check.
- Sort IDs lexicographically; stratified 60/20/20 split with seed 20261021 (sklearn train_test_split, 40% holdout then half). Check exact ID disjointness. Quantify identical feature vectors within/across splits, including conflicting labels. Do not claim unseen-pattern or temporal generalization.
- Pilot: stratified 30,000 blocks sampled solely from final training; internally split 60/20/20. No pilot block enters final validation/test. Same predeclared model settings; no hyperparameter search.
- Representations: event counts (frequency), and count>0 (presence). Same split for all six configurations.
- Rule: max_j abs((x_j - mean_normal_train_j) / max(std_normal_train_j, 1)). Fit normal training blocks only; population SD. The scale floor is fixed for stability, not optimized. This deliberately simple rule ignores feature interactions.
- Logistic Regression: log1p for frequency only, StandardScaler fitted on training; C=1, balanced class weights, lbfgs, max_iter=2000, random_state=20261021. No feature selection.
- Random Forest: 100 trees, max_depth=12, min_samples_leaf=5, max_features=sqrt, balanced_subsample class weights, n_jobs=2, seed 20261021. Raw counts or binary presence.
- Two thresholds selected using validation labels only: maximum F1; and maximum recall subject to validation FPR<=0.01. Ties prefer fewer FP, then higher threshold. Equal scores are inseparable; include an above-maximum score threshold (no positive predictions). Classification uses score>=threshold. Freeze all six threshold pairs before any final test scoring. No post-test model or threshold changes.
- Positive class is Anomaly. P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN), FPR=FP/(FP+TN). Undefined precision is blank/NaN, never invented. PR-AUC is trapezoidal sklearn auc(recall,precision); average precision (AP) is separately reported, not treated as synonymous. All rows including failures remain accounted for; stop on incomplete scoring.
- Actual test FPR is reported without enforcing the validation constraint on test. Scores, split, parameters and environment are retained locally for exact rescoring. Descriptive seen/unseen-feature and error-pattern tables are diagnostics; do not tune on them.
- This is one fixed split in one benchmark environment. No confidence or causal claims from correlated blocks; no attack labels, novel algorithm claim, online deployment claim, or retrospective human approval. Full rescoring and figure regeneration must agree with saved results.
