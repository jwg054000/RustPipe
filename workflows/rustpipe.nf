#!/usr/bin/env nextflow

// ─────────────────────────────────────────────────────────────────────────────
// RustPipe workflow
//
// Runs fast downstream RNA-seq analysis (TMM normalisation, DE, PCA, GSEA)
// on a count matrix, or on BAM+GTF via rustqc then convert.
//
// Typical usage:
//   nextflow run jwg054000/RustPipe -r main \
//       --input  counts.tsv \
//       --metadata samples.csv \
//       --group_col condition
//
// With gene-set enrichment:
//   nextflow run jwg054000/RustPipe -r main \
//       --input  counts.tsv \
//       --metadata samples.csv \
//       --group_col condition \
//       --gene_sets msigdb_hallmarks.gmt
// ─────────────────────────────────────────────────────────────────────────────

nextflow.enable.dsl = 2

include { RUSTPIPE } from '../modules/local/rustpipe/main'
include { RUSTQC_RNA; FEATURECOUNTS_CONVERT } from '../modules/local/rustqc_rna/main'

// ── Parameter validation ──────────────────────────────────────────────────────

if (!params.metadata) error "Please provide --metadata (path to samplesheet CSV)"
if (!params.group_col) error "Please provide --group_col (column name for contrasts)"
if (!params.bam && !params.input) error "Provide --input (counts TSV) or --bam plus --gtf"
if (params.bam && !params.gtf) error "Please provide --gtf with --bam"

// ── Workflow ──────────────────────────────────────────────────────────────────

workflow {

    // Build a minimal meta map so the module can tag log lines and outputs.
    def meta = [ id: params.group_col ]

    // Count matrix: either a pre-merged TSV, or rustqc featureCounts from BAMs.
    if (params.bam) {
        bam_ch = Channel.fromPath(params.bam)
            .map { f -> [ [id: f.baseName], f ] }
        RUSTQC_RNA(bam_ch, file(params.gtf))
        FEATURECOUNTS_CONVERT(RUSTQC_RNA.out.counts.collect())
        counts_ch = FEATURECOUNTS_CONVERT.out.tsv.map { f -> [ meta, f ] }
    } else {
        counts_ch = Channel.of( [ meta, file(params.input) ] )
    }

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
