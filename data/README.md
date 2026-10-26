# HDFS_v1 data

Official sources and licence are documented in [../docs/DATA_SOURCES.md](../docs/DATA_SOURCES.md). Research/academic use only under the supplied [notice](LOGHUB_LICENSE.txt); cite Loghub. No original dataset is included in the repository.

From the P3 directory, run `python scripts/acquire.py` after creating the documented environment. This downloads the official 186,645,559-byte ZIP, verifies the published MD5, calculates SHA-256, inspects the actual ZIP directory, and extracts four selected preprocessed members only. It rejects unsafe paths/symlinks/duplicate names and caps download/extraction sizes. It never executes files or unpickles NPZ arrays. At least 3 GB free space is required. An interrupted `.part` file is left in place; inspect it and remove only that incomplete file before retrying. No automatic bypass or alternate source is used.

- `manifest.json`: actual URLs, versions, sizes and hashes.
- `archive_inventory.json`: full archive member listing; raw HDFS.log stays compressed.
- `raw/`: read-only originals (ignored by Git).
- `processed/`: derived numeric arrays, deterministic split and per-model validation/test scores (ignored).
- `interim/`: acquisition/setup/validation logs and local frozen-project inventory (ignored).

`python scripts/prepare.py` validates hashes, IDs, labels, counts and all trace-to-matrix counts, then builds derived arrays. Existing files are verified on repeat acquisition; raw inputs are not overwritten. `Label`, `Type`, `BlockId`, trace timing and latency are not model features. Official HDFS.npz has object-valued sequences and no ID key, so only its NPY headers are inspected. Model inputs come exclusively from ID-keyed CSVs.
