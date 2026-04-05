process RUSTPIPE {
    tag "$meta.id"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/rustpipe:0.1.0' :
        'ghcr.io/jwg054000/RustPipe:0.1.0' }"

    input:
    // Count matrix produced by Salmon (quant.sf) or STAR (ReadsPerGene.out.tab),
    // normalised to a merged TSV by nf-core/rnaseq tximeta/tximport steps.
    tuple val(meta), path(counts)
    // CSV samplesheet with at minimum a sample_id column and the group column.
    path samplesheet
    // Name of the column in samplesheet that defines contrast groups (e.g. "condition").
    val  group_col
    // Optional GMT/JSON gene-set file for GSEA.  Pass file('NO_FILE') when unused.
    path gene_sets

    output:
    tuple val(meta), path("rustpipe_out/de/*.parquet"),       emit: de_results
    tuple val(meta), path("rustpipe_out/pca/*.parquet"),      emit: pca
    path "rustpipe_out/*_mqc.tsv",                            emit: multiqc_files
    tuple val(meta), path("rustpipe_out/enrichment/*.parquet"),  emit: enrichment,  optional: true
    path "rustpipe_out/pipeline_manifest.json",               emit: manifest
    path "versions.yml",                                       emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args        = task.ext.args ?: ''
    // Only pass --gene-sets when the user supplied a real file.
    def gs_arg      = gene_sets.name != 'NO_FILE' ? "--gene-sets ${gene_sets}" : ''
    // Let rustpipe saturate all cores allocated to the process by Nextflow.
    def threads_arg = task.cpus ? "--threads ${task.cpus}" : ''
    """
    rustpipe pipeline \\
        --input        ${counts}      \\
        --metadata     ${samplesheet} \\
        --group-col    ${group_col}   \\
        --output       rustpipe_out   \\
        ${gs_arg}      \\
        ${threads_arg} \\
        ${args}        \\
        -v

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        rustpipe: \$(rustpipe --version | sed 's/rustpipe //')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p rustpipe_out/{de,pca,enrichment}
    touch rustpipe_out/de/contrast.parquet
    touch rustpipe_out/pca/scores.parquet
    touch rustpipe_out/pca/variance.parquet
    touch rustpipe_out/pca_mqc.tsv
    touch rustpipe_out/sample_dists_mqc.tsv
    touch rustpipe_out/pipeline_manifest.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        rustpipe: \$(rustpipe --version | sed 's/rustpipe //')
    END_VERSIONS
    """
}
