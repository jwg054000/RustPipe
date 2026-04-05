# Changelog

All notable changes to RustPipe will be documented in this file.

## [0.1.0] - 2026-04-04

### Added
- TMM normalization matching edgeR (Robinson & Oshlack 2010)
- Differential expression: Welch's t-test (bulk) and Wilcoxon rank-sum (single-cell)
- PCA via randomized SVD with double-QR power iteration (Halko et al. 2011)
- Preranked GSEA with gene-label permutation and NES normalization (Subramanian et al. 2005)
- CSV to Parquet conversion
- Full pipeline orchestrator (normalize, DE, PCA, enrich)
- Built-in benchmark mode on synthetic data
- Parquet I/O throughout
- Automatic multi-core parallelism
- 14 unit tests + 6 integration tests

### Validated against
- edgeR 4.6.3 (TMM: log2 CPM r = 0.9998)
- R Welch t.test (t-statistic Spearman rho = 0.999)
- R prcomp (PC1-3 correlation > 0.984)
- fgsea (NES Spearman rho = 0.9993, 50/50 direction agreement)
- Tested on TCGA-BRCA (19,938 genes x 754 samples), PBMC 10K, and Visium spatial data
