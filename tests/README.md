# Tests

Run `python -m unittest discover -s tests -v` from the P3 root. Tests cover keyed joining despite reordered rows, forbidden predictors, bad IDs/labels/counts, deterministic complete disjoint splits, known confusion counts, undefined precision, tied-score FPR constraints, normal-only rule fitting, int8 label overflow and AP versus trapezoidal PR-AUC.
