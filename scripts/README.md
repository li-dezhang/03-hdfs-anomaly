# Executable workflow

Run from the P3 root with the pinned environment. `acquire.py` downloads and verifies official data; `prepare.py` checks keyed joins/trace counts and creates splits; `run.py` fits six fixed configurations, freezes validation thresholds, then scores test; `evaluate.py` independently rescores saved predictions, verifies the exported metrics, and generates figures from CSV. Run IDs must be new for training; existing experiment directories are not overwritten. See the main README for tested commands.
