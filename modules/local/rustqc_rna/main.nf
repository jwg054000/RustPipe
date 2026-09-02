process RUSTQC_RNA {
    tag "$meta.id"
    label 'process_medium'

    container 'ghcr.io/seqeralabs/rustqc:v0.2.1@sha256:2d06741730318d4a419a91e6cc6c57c589759eb7ec1b2c0e74e0e1366c301189'

    input:
    tuple val(meta), path(bam)
    path gtf

    output:
    tuple val(meta), path("${meta.id}_qc"), emit: qcdir
    path "${meta.id}_qc/featurecounts/*.featureCounts.tsv", emit: counts
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def paired = params.paired ? '--paired' : ''
    """
    rustqc rna ${bam} --gtf ${gtf} ${paired} --outdir ${meta.id}_qc

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        rustqc: \$(rustqc --version 2>/dev/null | head -n1 || echo unknown)
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${meta.id}_qc/featurecounts
    echo -e "Geneid\tChr\tStart\tEnd\tStrand\tLength\t${meta.id}\nG1\t1\t1\t2\t+\t2\t0" > ${meta.id}_qc/featurecounts/${meta.id}.featureCounts.tsv
    echo '"${task.process}":\n    rustqc: stub' > versions.yml
    """
}

process FEATURECOUNTS_CONVERT {
    tag "convert"
    label 'process_medium'

    conda "${projectDir}/modules/local/rustpipe/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/rustpipe:0.1.0' :
        'ghcr.io/jwg054000/rustpipe:0.1.0@sha256:16ba09adbf9085beecd9e10fc57aa6a12f019f1a0f50ce5ddc42c91fdb3cb3bf' }"

    input:
    path tsvs

    output:
    path "counts.parquet", emit: parquet
    path "counts.tsv", emit: tsv

    script:
    """
    python3 - << 'PY'
import csv, glob, os
ann = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
genes = []
seen = set()
tables = []
paths = sorted(glob.glob("*.tsv") + glob.glob("*.featureCounts.tsv"))
uniq = []
for p in paths:
    if p not in uniq:
        uniq.append(p)
for path in uniq:
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    rdr = csv.DictReader(lines, delimiter="\\t")
    if not rdr.fieldnames:
        raise SystemExit("empty featureCounts: " + path)
    count_cols = [c for c in rdr.fieldnames if c not in ann]
    if not count_cols:
        raise SystemExit("no sample columns in " + path)
    sample = os.path.splitext(os.path.basename(path))[0].replace(".featureCounts", "")
    col = count_cols[0]
    rows = {}
    for rec in rdr:
        gid = rec["Geneid"]
        rows[gid] = rec[col]
        if gid not in seen:
            genes.append(gid)
            seen.add(gid)
    tables.append((sample, rows))
with open("counts.tsv", "w") as out:
    out.write("gene\\t" + "\\t".join(s for s, _ in tables) + "\\n")
    for g in genes:
        out.write(g + "\\t" + "\\t".join(t.get(g, "0") for _, t in tables) + "\\n")
PY
    rustpipe convert -i counts.tsv -o counts.parquet
    """
}
