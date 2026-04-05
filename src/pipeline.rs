//! Full pipeline orchestrator and benchmark mode.
//!
//! Chains: [filter] → normalize → DE → PCA → enrich in one command.
//! Also provides synthetic-data benchmarks for performance testing.

use crate::convert;
use crate::de;
use crate::enrich;
use crate::filter;
use crate::hvg;
use crate::io;
use crate::normalize;
use crate::pca;
use anyhow::Result;
use log::info;
use polars::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::path::Path;
use std::time::Instant;

/// Run the full pipeline: [filter] → normalize → [HVG] → DE → PCA → enrich.
#[allow(clippy::too_many_arguments)]
pub fn run_full(
    input: &Path,
    metadata: &Path,
    group_col: &str,
    output_dir: &Path,
    gene_sets: Option<&Path>,
    n_pcs: usize,
    skip_normalize: bool,
    n_hvg: Option<usize>,
    min_count: Option<f64>,
    min_samples: Option<usize>,
    seed: u64,
) -> Result<()> {
    let start = Instant::now();
    std::fs::create_dir_all(output_dir)?;

    // Auto-convert TSV/CSV input to Parquet if needed
    let input = {
        let input_str = input.to_string_lossy().to_lowercase();
        let is_tabular = input_str.ends_with(".tsv")
            || input_str.ends_with(".csv")
            || input_str.ends_with(".txt")
            || input_str.ends_with(".tsv.gz")
            || input_str.ends_with(".csv.gz");
        if is_tabular {
            let converted = output_dir.join("_input_converted.parquet");
            info!("Auto-converting tabular input to Parquet: {}", input.display());
            convert::csv_to_parquet(input, &converted, 3, None)?;
            converted
        } else {
            input.to_path_buf()
        }
    };

    // Auto-convert TSV/CSV metadata to Parquet if needed
    let metadata = {
        let meta_str = metadata.to_string_lossy().to_lowercase();
        let is_tabular = meta_str.ends_with(".tsv")
            || meta_str.ends_with(".csv")
            || meta_str.ends_with(".txt")
            || meta_str.ends_with(".tsv.gz")
            || meta_str.ends_with(".csv.gz");
        if is_tabular {
            let converted = output_dir.join("_metadata_converted.parquet");
            info!("Auto-converting tabular metadata to Parquet: {}", metadata.display());
            // Detect separator: comma for .csv, tab for .tsv/.txt
            let sep = if meta_str.ends_with(".csv") || meta_str.ends_with(".csv.gz") {
                b','
            } else {
                b'\t'
            };
            let mut df = LazyCsvReader::new(metadata.to_string_lossy().to_string())
                .with_has_header(true)
                .with_separator(sep)
                .with_infer_schema_length(Some(1000))
                .finish()?
                .collect()?;
            let file = std::fs::File::create(&converted)?;
            ParquetWriter::new(file).finish(&mut df)?;
            converted
        } else {
            metadata.to_path_buf()
        }
    };

    let input = input.as_path();
    let metadata = metadata.as_path();

    // Compute total steps dynamically
    let has_filter = min_count.is_some();
    let has_hvg = n_hvg.is_some();
    let has_enrich = gene_sets.is_some();
    let total_steps = 1 // normalize (or skip)
        + if has_filter { 1 } else { 0 }
        + if has_hvg { 1 } else { 0 }
        + 1 // DE
        + 1 // PCA
        + if has_enrich { 1 } else { 0 };
    let mut step = 1;
    let mut timings: Vec<(&str, f64)> = Vec::new();

    // Step: Filter low-count genes (optional, BEFORE normalization)
    let filter_input = if has_filter {
        let mc = min_count.unwrap();
        let ms = min_samples.unwrap_or(3);
        info!(
            "Step {}/{}: Low-count gene filtering (min_count={}, min_samples={})",
            step, total_steps, mc, ms
        );
        let step_start = Instant::now();
        let filtered_path = output_dir.join("filtered.parquet");
        filter::run_filter(input, &filtered_path, mc, ms)?;
        timings.push(("filter", step_start.elapsed().as_secs_f64()));
        step += 1;
        filtered_path
    } else {
        input.to_path_buf()
    };

    // Step: Normalize (unless already normalized)
    let norm_path = output_dir.join("normalized.parquet");
    let expr_input = if skip_normalize {
        info!("Skipping normalization (--skip-normalize)");
        filter_input
    } else {
        info!(
            "Step {}/{}: TMM normalization + log2 CPM",
            step, total_steps
        );
        let step_start = Instant::now();
        normalize::tmm_normalize(&filter_input, &norm_path, 1.0)?;
        timings.push(("normalize", step_start.elapsed().as_secs_f64()));
        step += 1;
        norm_path.clone()
    };

    // Step (optional): HVG selection after normalization, before DE/PCA
    let analysis_input = if let Some(n_top) = n_hvg {
        info!(
            "Step {}/{}: HVG selection (top {} genes)",
            step, total_steps, n_top
        );
        let step_start = Instant::now();
        let hvg_path = output_dir.join("hvg_filtered.parquet");
        hvg::run_hvg(&expr_input, &hvg_path, n_top)?;
        timings.push(("hvg", step_start.elapsed().as_secs_f64()));
        step += 1;
        hvg_path
    } else {
        expr_input.clone()
    };

    // Step: Differential expression (always on full expression, not HVG-filtered)
    info!(
        "Step {}/{}: Differential expression (Welch t-test)",
        step, total_steps
    );
    let step_start = Instant::now();
    let de_dir = output_dir.join("de");
    de::run_de(&expr_input, metadata, group_col, &de_dir, "welch")?;
    timings.push(("de", step_start.elapsed().as_secs_f64()));
    step += 1;

    // Step: PCA (on HVG-filtered if available, else full)
    info!(
        "Step {}/{}: PCA ({} components, 3 power iterations)",
        step, total_steps, n_pcs
    );
    let step_start = Instant::now();
    let pca_dir = output_dir.join("pca");
    let pca_result = pca::run_pca(&analysis_input, &pca_dir, n_pcs, 3)?;
    timings.push(("pca", step_start.elapsed().as_secs_f64()));
    step += 1;

    // Write MultiQC-compatible PCA outputs to output_dir (not pca subdir)
    pca::write_multiqc_pca(&pca_result.variance_explained, output_dir)?;
    pca::write_multiqc_dists(
        &pca_result.scores,
        &pca_result.sample_names,
        pca_result.n_samples,
        pca_result.n_pcs,
        output_dir,
    )?;
    info!("Wrote MultiQC PCA files: pca_mqc.tsv, sample_dists_mqc.tsv");

    // Step: Enrichment (optional)
    if let Some(gs_path) = gene_sets {
        info!("Step {}/{}: Gene set enrichment", step, total_steps);
        let step_start = Instant::now();
        let enrich_dir = output_dir.join("enrichment");
        enrich::run_enrichment(&de_dir, gs_path, &enrich_dir, 1000, seed)?;
        timings.push(("enrichment", step_start.elapsed().as_secs_f64()));
    } else {
        info!("Skipping enrichment (no gene sets provided)");
    }

    // Write pipeline manifest JSON
    let timings_json: Vec<serde_json::Value> = timings
        .iter()
        .map(|(name, secs)| {
            serde_json::json!({
                "step": name,
                "seconds": secs,
            })
        })
        .collect();

    let timestamp_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let gene_sets_str = gene_sets.map(|p| p.display().to_string());

    let manifest = serde_json::json!({
        "rustpipe_version": env!("CARGO_PKG_VERSION"),
        "timestamp": timestamp_secs,
        "inputs": {
            "expression": input.display().to_string(),
            "metadata": metadata.display().to_string(),
            "gene_sets": gene_sets_str,
        },
        "parameters": {
            "group_col": group_col,
            "n_pcs": n_pcs,
            "n_hvg": n_hvg,
            "min_count": min_count,
            "min_samples": min_samples,
            "skip_normalize": skip_normalize,
            "seed": seed,
        },
        "timings": timings_json,
        "total_seconds": start.elapsed().as_secs_f64(),
    });
    std::fs::write(
        output_dir.join("pipeline_manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    info!("Wrote pipeline_manifest.json");

    info!(
        "Full pipeline completed in {:.3}s",
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Generate synthetic expression data and benchmark all pipeline stages.
///
/// Creates a realistic count matrix with:
/// - Log-normal expression distribution (mimics RNA-seq)
/// - Two sample groups with differential expression in 5% of genes
/// - Library size variation across samples
pub fn bench(n_genes: usize, n_samples: usize, output_dir: &Path) -> Result<()> {
    let total_start = Instant::now();
    std::fs::create_dir_all(output_dir)?;

    info!("Generating synthetic {}×{} matrix...", n_genes, n_samples);
    let gen_start = Instant::now();

    let mut rng = StdRng::seed_from_u64(42);

    // Two groups of equal size
    let n_per_group = n_samples / 2;

    // Generate count matrix
    let mut matrix = vec![0.0f64; n_genes * n_samples];
    let n_de = n_genes / 20; // 5% DE genes

    // Library size variation
    let lib_scales: Vec<f64> = (0..n_samples)
        .map(|_| (1.0 + rng.sample::<f64, _>(StandardNormal) * 0.3).max(0.5))
        .collect();

    for g in 0..n_genes {
        let base_mean: f64 = (rng.sample::<f64, _>(StandardNormal) * 2.0 + 6.0).exp();

        for s in 0..n_samples {
            let group_effect = if g < n_de && s >= n_per_group {
                2.0 // 2x fold change for DE genes in group B
            } else {
                1.0
            };

            let lambda = base_mean * lib_scales[s] * group_effect;
            // Poisson-like with NB overdispersion
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            let count = (lambda * (1.0 + noise * 0.3).max(0.1)).max(0.0).round();
            matrix[g * n_samples + s] = count;
        }
    }

    info!("  Generated in {:.3}s", gen_start.elapsed().as_secs_f64());

    // Write raw counts
    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("SAMPLE{:04}", s)).collect();

    let raw_path = output_dir.join("raw_counts.parquet");
    io::write_expression_parquet(
        &raw_path,
        &gene_names,
        &sample_names,
        &matrix,
        n_genes,
        n_samples,
        3,
    )?;

    // Write metadata
    let meta_path = output_dir.join("metadata.parquet");
    {
        let ids = polars::prelude::Series::new("sample_id".into(), &sample_names);
        let groups: Vec<String> = (0..n_samples)
            .map(|s| {
                if s < n_per_group {
                    "GroupA".to_string()
                } else {
                    "GroupB".to_string()
                }
            })
            .collect();
        let grp = polars::prelude::Series::new("condition".into(), &groups);
        let df = polars::prelude::DataFrame::new(vec![ids, grp])?;
        let file = std::fs::File::create(&meta_path)?;
        polars::prelude::ParquetWriter::new(file).finish(&mut df.clone())?;
    }

    // Benchmark: Normalize
    info!("Benchmark: TMM normalization...");
    let norm_start = Instant::now();
    let norm_path = output_dir.join("normalized.parquet");
    normalize::tmm_normalize(&raw_path, &norm_path, 1.0)?;
    let norm_time = norm_start.elapsed().as_secs_f64();
    info!("  Normalize: {:.3}s", norm_time);

    // Benchmark: DE
    info!("Benchmark: Differential expression...");
    let de_start = Instant::now();
    let de_dir = output_dir.join("de");
    de::run_de(&norm_path, &meta_path, "condition", &de_dir, "welch")?;
    let de_time = de_start.elapsed().as_secs_f64();
    info!("  DE: {:.3}s", de_time);

    // Benchmark: PCA
    info!("Benchmark: PCA (50 components, 3 power iterations)...");
    let pca_start = Instant::now();
    let pca_dir = output_dir.join("pca");
    let _pca_result = pca::run_pca(&norm_path, &pca_dir, 50, 3)?;
    let pca_time = pca_start.elapsed().as_secs_f64();
    info!("  PCA: {:.3}s", pca_time);

    let total_time = total_start.elapsed().as_secs_f64();

    // Summary
    info!("═══════════════════════════════════════════════════");
    info!(
        "BENCHMARK RESULTS: {} genes × {} samples",
        n_genes, n_samples
    );
    info!("═══════════════════════════════════════════════════");
    info!(
        "  Data generation:  {:.3}s",
        gen_start.elapsed().as_secs_f64()
    );
    info!("  TMM normalize:    {:.3}s", norm_time);
    info!("  Differential expr: {:.3}s", de_time);
    info!("  PCA (rSVD):       {:.3}s", pca_time);
    info!("  ─────────────────────────────");
    info!("  Total:            {:.3}s", total_time);
    info!("═══════════════════════════════════════════════════");

    // Write timing results as JSON
    let timing = serde_json::json!({
        "n_genes": n_genes,
        "n_samples": n_samples,
        "normalize_seconds": norm_time,
        "de_seconds": de_time,
        "pca_seconds": pca_time,
        "total_seconds": total_time,
    });
    let timing_path = output_dir.join("benchmark_timing.json");
    std::fs::write(&timing_path, serde_json::to_string_pretty(&timing)?)?;

    Ok(())
}
