# P3 sources — verified 2026-10-21

This replaces the P3-local copy of the old five-project planning ledger; that copy is retained in history/DATA_SOURCES_PHASE0.md. Other projects have not changed.

- Official repository: https://github.com/logpai/loghub
- HDFS description: https://github.com/logpai/loghub/blob/master/HDFS/README.md
- Distribution: https://zenodo.org/records/8196385 (record version v8; dataset HDFS_v1)
- Actual ZIP: https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1
- Licence: https://github.com/logpai/loghub/blob/master/LICENSE

Archive: 186,645,559 bytes; publisher MD5 `76a24b4d9a6164d543fb275f89773260` verified. Locally computed SHA-256 `04f919f2185821f23f045dca611a7586429bdabc601bd7b43f30005f8e289b01`. Every selected file is hashed in ../data/manifest.json. All ZIP members and decompressed sizes are in ../data/archive_inventory.json. The raw log member is 1,577,982,906 bytes and remains unextracted. Published line counts and duration are source statements, not measurements of the raw log by this study.

Content verified: 575,061 unique blocks; 558,223 normal, 16,838 anomaly; 29 event columns. Labels and event traces match by ID, with zero count mismatches across all traces. NPZ contains object-valued x_data and int64 y_data but no Block IDs; it is not used or unpickled. Full validation and duplicate diagnostics are in ../results/data/.

Licence permits research/academic use, requires referencing the Loghub repository and applicable Loghub paper, and retaining the notice in dataset copies. It is not represented as MIT/Apache/CC0 or commercial-use permission. No dataset redistribution is planned. The exact notice is retained in ../data/LOGHUB_LICENSE.txt and with local raw files. Official README asks users to cite Zhu et al. (ISSRE 2023) and Jiang et al. (ISSTA 2024); the report includes both, while clearly stating that Loghub-2.0 is not the experimental dataset.

Unverified: original raw-line sharing across blocks; independence of physical incidents; cross-system and temporal generalization; independent verification of the corpus labels; train-only construction of upstream templates. A checked hash confirms file identity, not these scientific properties.
