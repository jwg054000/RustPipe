#!/usr/bin/env nextflow

// ─────────────────────────────────────────────────────────────────────────────
// RustPipe workflow
//
// Runs fast downstream RNA-seq analysis (TMM normalisation, DE, PCA, GSEA)
// on a count matrix from Salmon or STAR.
//
// Typical usage:
//   nextflow run prairiebio/rustpipe -r main \
//       --input  counts.tsv \
//       --metadata samples.csv \
//       --group_col condition
//
// With gene-set enrichment:
//   nextflow run prairiebio/rustpipe -r main \
//       --input  counts.tsv \
//       --metadata samples.csv \
//       --group_col condition \
//       --gene_sets msigdb_hallmarks.gmt
// ─────────────────────────────────────────────────────────────────────────────

nextflow.enable.dsl = 2

include { RUSTPIPE } from '../modules/local/rustpipe/main'

// ── Parameter validation ──────────────────────────────────────────────────────

if (!params.input)    error "Please provide --input (path to count matrix TSV)"
if (!params.metadata) error "Please provide --metadata (path to samplesheet CSV)"
if (!params.group_col) error "Please provide --group_col (column name for contrasts)"

// ── Workflow ──────────────────────────────────────────────────────────────────

workflow {

    // Build a minimal meta map so the module can tag log lines and outputs.
    def meta = [ id: params.group_col ]

    // Count matrix channel: single file wrapped with the meta map.
    counts_ch = Channel.of( [ meta, file(params.input) ] )

    // Samplesheet channel: plain file (not tuple).
    samplesheet_ch = Channel.value( file(params.metadata) )

    // Gene sets: use the NO_FILE sentinel when the user did not supply one.
    gene_sets_ch = params.gene_sets
        ? Channel.value( file(params.gene_sets) )
        : Channel.value( file('NO_FILE') )

    // ── Run RustPipe ─────────────────────────────────────────────────────────
    RUSTPIPE (
        counts_ch,
        samplesheet_ch,
        params.group_col,
        gene_sets_ch
    )

    // ── Collect outputs ───────────────────────────────────────────────────────
    // Results land in params.outdir via the publishDir directive in nextflow.config.
    // Versions are always emitted so the run is reproducible.
    RUSTPIPE.out.versions
        .collectFile(name: 'versions.yml', storeDir: params.outdir)
}
