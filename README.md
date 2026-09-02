# RustPipe

**Fast RNA-seq analysis in Rust.** 16x faster than R/edgeR, validated to r > 0.999 on real data.

Takes a count matrix and produces differential expression, PCA, and pathway enrichment results in seconds. The downstream companion to RustQC.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

## Performance

| Step | R/edgeR | RustPipe | Speedup |
|------|--------:|---------:|--------:|
| TMM + log2 CPM | 12 s | 1.8 s | 7x |
| Differential expression | 14 s | 0.14 s | 100x |
| PCA (50 PCs, rSVD) | 23 s | 1.1 s | 21x |
| GSEA (1K permutations) | 8 s | 0.24 s | 33x |
| HVG selection | 3.5 s | 0.37 s | 9x |
| **Full pipeline** | **49 s** | **3.0 s** | **16x** |

TCGA-BRCA, 19,938 genes x 754 samples (Basal vs Luminal A), Apple M-series.

## Validation

Every algorithm is validated against its R reference implementation on real-world datasets:

| Component | Reference | Metric | Agreement |
|-----------|-----------|--------|-----------|
| TMM normalization | edgeR 4.6.3 | Pearson r (log2 CPM) | 0.9998 |
| Differential expression | Welch t-test (R) | Spearman rho (t-statistics) | 0.999 |
| | | Jaccard (significant genes, FDR < 0.05) | 0.985 |
| PCA | prcomp (R) | Pearson r (PC1-3 scores) | 0.999, 0.999, 0.984 |
| GSEA | fgsea (R) | Spearman rho (NES, 50 pathways) | 0.9993 |
| HVG selection | Scanpy (Seurat v3) | Spearman rho (gene ranks) | 0.990 |
| | | Jaccard (top 500 HVGs) | 1.000 |

47 tests (35 unit + 12 integration).

## Install

```bash
# Clone and build (requires Rust 1.75+)
git clone https://github.com/jwg054000/RustPipe.git
cd RustPipe
cargo build --release

# Verify
./target/release/rustpipe --help
```

Don't have Rust? Install it from [rustup.rs](https://rustup.rs/).

## Container

Image name: `ghcr.io/jwg054000/rustpipe` (linux/amd64, built from this Dockerfile). prairie-rna-stream consumes the published digest-pinned image — do not copy this tree into that repo.

```bash
# After the first GHCR publish (Actions workflow or the commands below):
docker pull ghcr.io/jwg054000/rustpipe:0.1.0@sha256:16ba09adbf9085beecd9e10fc57aa6a12f019f1a0f50ce5ddc42c91fdb3cb3bf
# tags latest / 0.1.0 / sha-88c010a95cd1fa7d9e1dd6b425b6cb2c2054acfd share that digest
docker buildx imagetools inspect ghcr.io/jwg054000/rustpipe:0.1.0 --format '{{.Manifest.Digest}}'
```

Local publish (PAT needs `write:packages` and `read:packages`; do not store the token in git):

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u jwg054000 --password-stdin
docker build --platform linux/amd64 -t ghcr.io/jwg054000/rustpipe:0.1.0 \
  -t ghcr.io/jwg054000/rustpipe:$(git rev-parse HEAD) \
  -t ghcr.io/jwg054000/rustpipe:latest .
docker push ghcr.io/jwg054000/rustpipe:0.1.0
docker push ghcr.io/jwg054000/rustpipe:$(git rev-parse HEAD)
docker push ghcr.io/jwg054000/rustpipe:latest
docker buildx imagetools inspect ghcr.io/jwg054000/rustpipe:0.1.0
```

After the first push, set the GitHub package public (Packages → rustpipe → Package settings → Change visibility) so Nextflow can pull without auth.

## Quick Start

```bash
# Run the full pipeline in one command (accepts TSV, CSV, or Parquet)
rustpipe pipeline \
    -i counts.tsv \
    -m samples.csv \
    -g condition \
    -o results/ \
    --gene-sets hallmark_gene_sets.gmt \
    -v

# Or run steps individually:
rustpipe convert   -i expression.csv -o expression.parquet
rustpipe filter    -i expression.parquet -o filtered.parquet --min-count 10
rustpipe normalize -i filtered.parquet -o normalized.parquet
rustpipe hvg       -i normalized.parquet -o hvg.parquet --n-top-genes 2000
rustpipe de        -i normalized.parquet -m metadata.parquet -g condition -o de/ --method moderated
rustpipe pca       -i normalized.parquet -o pca/ --n-pcs 50
rustpipe enrich    -d de/ -g hallmark.gmt -o enrichment/
```

## Subcommands

| Command | Description |
|---------|-------------|
| `convert` | CSV/TSV to Parquet (auto-detects delimiter, handles gzip). `--from featurecounts` strips the 6 Subread annotation columns |
| `filter` | Low-count gene filtering (edgeR filterByExpr pattern) |
| `normalize` | TMM normalization + log2 CPM transform |
| `hvg` | Highly variable gene selection (VST, log-log OLS) |
| `de` | Differential expression (Welch, Wilcoxon, or moderated t) |
| `pca` | Principal component analysis (randomized SVD) |
| `enrich` | Gene set enrichment (permutation-based GSEA, GMT or JSON) |
| `pipeline` | Full pipeline: [filter] → normalize → [HVG] → DE → PCA → [enrich] |
| `bench` | Performance benchmarks on synthetic data |

## Why RustPipe?

| | RustPipe | edgeR + limma | Scanpy |
|---|---|---|---|
| **Speed** (20K genes x 750 samples) | 3 s | 49 s | ~60 s |
| **Install** | Single binary | R + Bioconductor | Python + conda |
| **I/O format** | Parquet (columnar) | RData / CSV | HDF5 |
| **Input** | TSV, CSV, Parquet, gzip | CSV / RData | HDF5 / CSV |
| **Parallelism** | Automatic | Manual | Automatic (numba) |
| **Enrichment** | Built-in GSEA | Separate package | Separate package |

**Use RustPipe when** you have a count matrix and need fast DE + PCA + enrichment in a pipeline.

**Use edgeR/limma when** you need GLMs, interaction terms, or voom precision weights.

**Use Scanpy when** you need clustering, trajectory inference, or annotation.

## Data Format

**Expression matrix** (TSV, CSV, or Parquet): rows = genes (first column = gene symbol), columns = samples.

**Metadata** (CSV or Parquet): first column = sample IDs matching the expression columns, remaining columns = group annotations.

**Gene sets** (GMT or JSON):
- GMT: MSigDB tab-separated format (`SET_NAME\tdescription\tGENE1\tGENE2\t...`)
- JSON: `{"SET_NAME": ["GENE1", "GENE2", ...], ...}`

The `pipeline` subcommand auto-detects all input formats.

## Methods

### TMM Normalization
Robinson & Oshlack (2010) trimmed mean of M-values (30% M-trim, 5% A-trim), matching edgeR's `calcNormFactors(method="TMM")`. Reference sample chosen by upper-quartile proximity.

### Low-count Gene Filtering
Matches edgeR `filterByExpr()` — keeps genes with at least `min_count` (default 10) in at least `min_samples` (default 3) samples. Runs before normalization.

### Highly Variable Gene Selection
Variance-stabilizing transform: fits ln(variance) = a + b·ln(mean) via OLS, computes standardized variance = observed/expected. Validated against Scanpy's Seurat v3 flavor (Spearman rho = 0.99, Jaccard = 0.91 at top 1000).

### Differential Expression
- **Welch's t-test** (default): unequal variance, Welch-Satterthwaite df
- **Moderated t** (`--method moderated`): limma-style empirical Bayes variance shrinkage (Smyth 2004) — stabilizes variance estimates for small sample sizes
- **Wilcoxon rank-sum** (`--method wilcoxon`): for single-cell, normal approximation with continuity correction

All with Benjamini-Hochberg FDR correction.

### PCA
Randomized SVD via Halko et al. (2011) Algorithm 4.4 with double-QR power iteration. 3 power iterations by default. Deterministic (seeded RNG). Outputs MultiQC-compatible TSV for nf-core integration.

### Gene Set Enrichment
Preranked GSEA following Subramanian et al. (2005):
- Weighted KS running-sum statistic with signed t-statistic ranking
- Gene-label permutation for p-values (1,000 default, `--n-permutations`)
- NES normalization against separate positive/negative null distributions
- Phipson-Smyth correction (no zero p-values)
- Benjamini-Hochberg FDR correction
- Accepts both GMT (MSigDB) and JSON gene set formats

## nf-core Integration

RustPipe includes an nf-core module and can slot into nf-core/rnaseq after quantification:

```
STAR/SALMON → COUNT_MERGE → [RustPipe] → MULTIQC
```

```bash
# Standalone Nextflow workflow
nextflow run jwg054000/RustPipe \
    --input counts.tsv \
    --metadata samples.csv \
    --group_col condition \
    -profile docker
```

Outputs MultiQC-compatible files (`pca_mqc.tsv`, `sample_dists_mqc.tsv`) and a pipeline manifest with per-step timing.

## Benchmarks

```bash
# Default: 18,168 genes x 800 samples
rustpipe bench -o bench_output/ -v

# Custom size
rustpipe bench --n-genes 20000 --n-samples 1000 -o bench_output/ -v
```

## Roadmap

- [x] v0.1: TMM, DE (Welch + Wilcoxon), PCA (rSVD), GSEA (permutation)
- [x] v0.2: GMT gene sets, HVG selection, gene filtering, moderated t, nf-core module, TSV/gzip input, Parquet enrichment output
- [ ] v0.3: Sparse CSR for single-cell, Moran's I for spatial transcriptomics
- [ ] v0.4: Python bindings (PyO3), R bindings

## Contributing

PRs welcome. Please include tests:

```bash
cargo test
cargo clippy -- -D clippy::all
cargo fmt --check
```

## References

- Robinson MD, Oshlack A (2010). A scaling normalization method for differential expression analysis of RNA-seq data. *Genome Biology* 11:R25.
- Smyth GK (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *Stat. Appl. Genet. Mol. Biol.* 3(1).
- Halko N, Martinsson PG, Tropp JA (2011). Finding structure with randomness. *SIAM Review* 53(2):217-288.
- Subramanian A et al. (2005). Gene set enrichment analysis. *PNAS* 102(43):15545-15550.
- Phipson B, Smyth GK (2010). Permutation P-values should never be zero. *Stat. Appl. Genet. Mol. Biol.* 9(1).
- Stuart T et al. (2019). Comprehensive integration of single-cell data. *Cell* 177(7):1888-1902.

## License

GPL-3.0. See [LICENSE](LICENSE).

## Citation

```
Josh Garton, 2026. RustPipe: Fast downstream RNA-seq analysis in Rust.
https://github.com/jwg054000/RustPipe
```


## Seqera RustQC

RustPipe does not vendor RustQC. Pin and run the published image, then convert:

```
rustqc rna $bam --gtf $gtf --paired --outdir ${id}_qc
rustpipe convert --from featurecounts ${id}_qc/featurecounts/*.featureCounts.tsv -o counts.parquet
rustpipe pipeline -i counts.parquet -m samples.csv -g condition -o results/
```

Nextflow: `--bam` plus `--gtf` runs process `RUSTQC_RNA` then the existing `RUSTPIPE` process. Image pin: `ghcr.io/seqeralabs/rustqc:v0.2.1@sha256:2d06741730318d4a419a91e6cc6c57c589759eb7ec1b2c0e74e0e1366c301189`.
