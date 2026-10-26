# Research implementation

`hdfs_study/core.py` contains strict keyed data merging, deterministic splits, the normal-profile rule, metric calculations and tie-aware validation threshold selection. Models and orchestration are in `../scripts/run.py`; all preprocessing is fitted within its training subset. `../tests/test_core.py` checks scientific invariants and metric edge cases.
