# RustPipe

Downstream of Seqera RustQC. Consume rustqc as a published binary/image only — never a git submodule or fork of seqeralabs/RustQC.

- Counts contract: featureCounts TSV → gene × sample parquet (`rustpipe convert --from featurecounts`)
- Do not treat a BAM as the count matrix. `RUSTQC_RNA` runs rustqc; convert only sees the TSV
- Do not add scRNA here (that is RustPipe-SC)
- Do not change TMM / DE / PCA / GSEA math
- GPL-3.0. Prefer a PR over committing to main
- Git author and committer must be Josh Garton `<jwg054000@gmail.com>`. Never Grok, Cursor Agent, Copilot, or any other tool identity. No vendor PR footers.
