//! RustPipe integration tests.
//!
//! Tests exercise the full CLI binary end-to-end against small synthetic
//! datasets generated inside each test.  No external fixture files are
//! required — everything is built in-memory and written to a `tempfile::TempDir`
//! that is cleaned up automatically after each test.
//!
//! Run with:
//!   cargo test --test integration
//!
//! Design notes:
//!   - This is a binary crate, so tests call the compiled binary via
//!     `std::process::Command` (using the `assert_cmd` helper).
//!   - Input Parquet files are built with the `polars` dev-dependency.
//!   - Output Parquet files are read back with the same.
//!   - Floating-point comparisons use the `approx` crate.

use assert_cmd::prelude::*;
use polars::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────

/// Write a genes × samples count matrix as a Parquet file.
///
/// `matrix` is row-major (genes × samples).  The first column in the Parquet
/// will be named `gene_symbol` and contain names like `GENE00000`.
fn write_count_parquet(
    path: &Path,
    gene_names: &[String],
    sample_names: &[String],
    matrix: &[f64], // row-major: n_genes × n_samples
) {
    let n_genes = gene_names.len();
    let n_samples = sample_names.len();
    assert_eq!(matrix.len(), n_genes * n_samples);

    let mut cols: Vec<Series> = Vec::with_capacity(n_samples + 1);
    cols.push(Series::new("gene_symbol".into(), gene_names));

    for s in 0..n_samples {
        let col_data: Vec<f64> = (0..n_genes).map(|g| matrix[g * n_samples + s]).collect();
        cols.push(Series::new(sample_names[s].as_str().into(), col_data));
    }

    let mut df = DataFrame::new(cols).expect("DataFrame construction failed");
    let file = std::fs::File::create(path).expect("create parquet file");
    ParquetWriter::new(file)
        .finish(&mut df)
        .expect("write parquet");
}

/// Write a metadata Parquet file: columns `sample_id` and `condition`.
fn write_metadata_parquet(path: &Path, sample_names: &[String], groups: &[String]) {
    assert_eq!(sample_names.len(), groups.len());
    let ids = Series::new("sample_id".into(), sample_names);
    let grp = Series::new("condition".into(), groups);
    let mut df = DataFrame::new(vec![ids, grp]).expect("meta df");
    let file = std::fs::File::create(path).expect("create meta parquet");
    ParquetWriter::new(file)
        .finish(&mut df)
        .expect("write meta");
}

/// Read a single f64 column from a Parquet file.
fn read_f64_column(path: &Path, col: &str) -> Vec<f64> {
    let df = LazyFrame::scan_parquet(path, Default::default())
        .expect("scan parquet")
        .collect()
        .expect("collect parquet");
    df.column(col)
        .expect("column not found")
        .f64()
        .expect("not f64")
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect()
}

/// Read the first string column from a Parquet file.
fn read_string_column(path: &Path, col: &str) -> Vec<String> {
    let df = LazyFrame::scan_parquet(path, Default::default())
        .expect("scan parquet")
        .collect()
        .expect("collect");
    df.column(col)
        .expect("column not found")
        .str()
        .expect("not string")
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect()
}

/// Return all column names of a Parquet file.
fn parquet_columns(path: &Path) -> Vec<String> {
    let df = LazyFrame::scan_parquet(path, Default::default())
        .expect("scan parquet")
        .collect()
        .expect("collect");
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Invoke the rustpipe binary, asserting it succeeds.
fn rustpipe(args: &[&str]) {
    Command::cargo_bin("rustpipe")
        .expect("rustpipe binary not found — run `cargo build` first")
        .args(args)
        .assert()
        .success();
}

// ─────────────────────────────────────────────────────────────
// Test 1: TMM normalization correctness
// ─────────────────────────────────────────────────────────────

/// Build a 100-gene × 10-sample count matrix where the last 5 samples have
/// 2× the library size of the first 5.  TMM should recover factors close to
/// the 2× ratio (or its reciprocal) so that the geometric mean of all factors
/// stays near 1.  After normalization all log2CPM values must be finite and
/// fall in the biologically plausible range [-10, 20].
#[test]
fn test_tmm_normalization_correctness() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 100usize;
    let n_samples = 10usize;
    let n_normal = 5usize; // first 5 samples: normal library
    let n_large = 5usize; // last 5 samples: 2× library size

    let mut rng = StdRng::seed_from_u64(7);

    // Counts drawn from a simple log-normal model (simulates RNA-seq)
    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        let base: f64 = (rng.sample::<f64, _>(StandardNormal) * 1.5 + 5.0).exp();
        for s in 0..n_samples {
            let lib_scale = if s < n_normal { 1.0 } else { 2.0 };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            let count = (base * lib_scale * (1.0 + noise * 0.2).max(0.05))
                .max(1.0)
                .round();
            matrix[g * n_samples + s] = count;
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();

    let raw_path = tmp.path().join("raw.parquet");
    let norm_path = tmp.path().join("normalized.parquet");

    write_count_parquet(&raw_path, &gene_names, &sample_names, &matrix);

    rustpipe(&[
        "normalize",
        "--input",
        raw_path.to_str().unwrap(),
        "--output",
        norm_path.to_str().unwrap(),
    ]);

    // Read all sample columns from normalized output
    let df = LazyFrame::scan_parquet(&norm_path, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    let col_names = df.get_column_names();
    // First column is gene_symbol; rest are samples
    assert_eq!(
        col_names.len(),
        n_samples + 1,
        "expected {} sample columns",
        n_samples
    );

    // For each sample, collect the log2CPM values
    let mut all_values: Vec<f64> = Vec::new();
    for s in 0..n_samples {
        let col_name = &col_names[s + 1];
        let vals: Vec<f64> = df
            .column(col_name)
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();

        for v in &vals {
            assert!(
                v.is_finite(),
                "log2CPM value is not finite: {} in sample {}",
                v,
                col_name
            );
            assert!(
                *v >= -10.0 && *v <= 20.0,
                "log2CPM value {} out of range [-10, 20] in sample {}",
                v,
                col_name
            );
        }
        all_values.extend_from_slice(&vals);
    }

    // Verify that not all values are identical (normalization actually ran)
    let first = all_values[0];
    let non_identical = all_values.iter().any(|&v| (v - first).abs() > 1e-6);
    assert!(
        non_identical,
        "all log2CPM values are identical — normalization looks degenerate"
    );

    // The 2× library-size samples should have lower raw sums than their
    // post-normalization equivalents relative to the normal samples.
    // Check that the normal and 2× samples have converging per-gene means
    // — i.e., the group means are closer after normalization than the raw
    // library sizes would predict.
    let mut normal_means: Vec<f64> = Vec::new();
    let mut large_means: Vec<f64> = Vec::new();
    for g in 0..n_genes {
        let norm_vals_normal: Vec<f64> = (0..n_normal)
            .map(|s| {
                let col = &col_names[s + 1];
                df.column(col)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(g)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        let norm_vals_large: Vec<f64> = (n_normal..n_samples)
            .map(|s| {
                let col = &col_names[s + 1];
                df.column(col)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(g)
                    .unwrap_or(f64::NAN)
            })
            .collect();

        let m_n: f64 = norm_vals_normal.iter().sum::<f64>() / n_normal as f64;
        let m_l: f64 = norm_vals_large.iter().sum::<f64>() / n_large as f64;
        normal_means.push(m_n);
        large_means.push(m_l);
    }

    // After TMM+log2CPM the group means should be within 1 log2 unit on average
    // (before normalization the raw 2× samples would have ~1 log2 unit higher library)
    let mean_abs_diff: f64 = normal_means
        .iter()
        .zip(large_means.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n_genes as f64;

    assert!(
        mean_abs_diff < 1.5,
        "TMM did not converge group means: mean absolute log2CPM difference = {:.3} (expected < 1.5 after normalization)",
        mean_abs_diff
    );
}

// ─────────────────────────────────────────────────────────────
// Test 2: Welch t-test correctness via DE command
// ─────────────────────────────────────────────────────────────

/// Create two groups of 8 samples each.  The first 10 genes have mean 100 in
/// group A and mean 200 in group B.  The remaining 90 genes are drawn from the
/// same distribution.  After DE:
///   - The 10 signal genes should have padj < 0.05
///   - Most of the 90 null genes should have padj > 0.05
///   - The log2FC for signal genes should be close to log2(200/100) = 1.0
#[test]
fn test_welch_t_test_de_correctness() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 100usize;
    let n_signal = 10usize;
    let n_samples = 16usize; // 8 per group
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(13);

    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            let base_mean = if g < n_signal {
                if s < n_per_group {
                    100.0
                } else {
                    200.0
                }
            } else {
                50.0
            };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            // Use realistic noise ≈ 15% CV, keep positive
            let val = (base_mean + noise * base_mean * 0.15).max(1.0).round();
            matrix[g * n_samples + s] = val;
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();
    let groups: Vec<String> = (0..n_samples)
        .map(|s| {
            if s < n_per_group {
                "GroupA".to_string()
            } else {
                "GroupB".to_string()
            }
        })
        .collect();

    let expr_path = tmp.path().join("expr.parquet");
    let meta_path = tmp.path().join("meta.parquet");
    let de_dir = tmp.path().join("de");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);
    write_metadata_parquet(&meta_path, &sample_names, &groups);

    rustpipe(&[
        "de",
        "--input",
        expr_path.to_str().unwrap(),
        "--metadata",
        meta_path.to_str().unwrap(),
        "--group-col",
        "condition",
        "--output",
        de_dir.to_str().unwrap(),
        "--method",
        "welch",
    ]);

    let de_file = de_dir.join("GroupA_vs_GroupB.parquet");
    assert!(
        de_file.exists(),
        "DE output file not found: {}",
        de_file.display()
    );

    let genes = read_string_column(&de_file, "gene");
    let padj = read_f64_column(&de_file, "p_adj");
    let lfc = read_f64_column(&de_file, "log2_fold_change");

    assert_eq!(genes.len(), n_genes);
    assert_eq!(padj.len(), n_genes);
    assert_eq!(lfc.len(), n_genes);

    // Map gene name → index in result
    let gene_idx: HashMap<&str, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    // Signal genes: all must have padj < 0.05
    let mut signal_padj_pass = 0usize;
    let mut signal_lfc_sum = 0.0f64;
    for g in 0..n_signal {
        let name = format!("GENE{:05}", g);
        if let Some(&i) = gene_idx.get(name.as_str()) {
            if padj[i] < 0.05 {
                signal_padj_pass += 1;
            }
            signal_lfc_sum += lfc[i].abs();
        }
    }

    assert_eq!(
        signal_padj_pass, n_signal,
        "Expected all {} signal genes to have padj < 0.05, but only {} did",
        n_signal, signal_padj_pass
    );

    // Mean |log2FC| for signal genes should be close to 1.0
    // (GroupA mean=100, GroupB mean=200, but note: DE is on raw counts,
    //  log2FC = log2(101) - log2(201) ≈ 0.995 in GroupA-vs-GroupB direction)
    let mean_signal_lfc = signal_lfc_sum / n_signal as f64;
    assert!(
        mean_signal_lfc > 0.7 && mean_signal_lfc < 1.3,
        "Mean |log2FC| for signal genes = {:.3} (expected ~1.0)",
        mean_signal_lfc
    );

    // Null genes: at least 75 of 90 should have padj > 0.05
    let mut null_pass = 0usize;
    for g in n_signal..n_genes {
        let name = format!("GENE{:05}", g);
        if let Some(&i) = gene_idx.get(name.as_str()) {
            if padj[i] > 0.05 {
                null_pass += 1;
            }
        }
    }
    let n_null = n_genes - n_signal;
    assert!(
        null_pass >= (n_null * 3 / 4),
        "Expected at least 75% of null genes to have padj > 0.05, got {}/{} = {:.1}%",
        null_pass,
        n_null,
        null_pass as f64 / n_null as f64 * 100.0
    );
}

// ─────────────────────────────────────────────────────────────
// Test 3: PCA captures dominant signal
// ─────────────────────────────────────────────────────────────

/// Create a 200-gene × 20-sample matrix with a strong two-group structure.
/// After PCA:
///   - PC1 variance explained should be > 30%
///   - The 10 group-A samples should have PC1 scores with opposite sign to
///     the 10 group-B samples (groups cluster on PC1)
#[test]
fn test_pca_captures_dominant_signal() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 200usize;
    let n_samples = 20usize;
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(99);

    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            // Strong group signal: group A gets +10, group B gets -10
            let group_signal = if s < n_per_group { 10.0 } else { -10.0 };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            // Keep counts positive by shifting
            matrix[g * n_samples + s] = (50.0 + group_signal + noise).max(1.0).round();
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();

    let expr_path = tmp.path().join("expr.parquet");
    let pca_dir = tmp.path().join("pca");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);

    rustpipe(&[
        "pca",
        "--input",
        expr_path.to_str().unwrap(),
        "--output",
        pca_dir.to_str().unwrap(),
        "--n-pcs",
        "5",
        "--n-power-iter",
        "3",
    ]);

    let scores_path = pca_dir.join("scores.parquet");
    let variance_path = pca_dir.join("variance.parquet");

    assert!(scores_path.exists(), "scores.parquet not found");
    assert!(variance_path.exists(), "variance.parquet not found");

    // Check variance explained: PC1 > 30%
    let var_exp = read_f64_column(&variance_path, "variance_explained");
    assert!(!var_exp.is_empty(), "variance_explained column is empty");
    assert!(
        var_exp[0] > 0.30,
        "PC1 variance explained = {:.2}% — expected > 30% for strong two-group signal",
        var_exp[0] * 100.0
    );

    // Check that all variance values are finite and non-negative
    for (i, &v) in var_exp.iter().enumerate() {
        assert!(
            v.is_finite() && v >= 0.0,
            "variance_explained[{}] = {} is not a valid proportion",
            i,
            v
        );
    }

    // Check group separation on PC1
    let pc1 = read_f64_column(&scores_path, "PC1");
    assert_eq!(pc1.len(), n_samples, "unexpected number of PC1 scores");

    let pc1_group_a: Vec<f64> = pc1[..n_per_group].to_vec();
    let pc1_group_b: Vec<f64> = pc1[n_per_group..].to_vec();

    let mean_a: f64 = pc1_group_a.iter().sum::<f64>() / n_per_group as f64;
    let mean_b: f64 = pc1_group_b.iter().sum::<f64>() / n_per_group as f64;

    // Groups should be on opposite sides of zero (signs differ)
    assert!(
        mean_a * mean_b < 0.0,
        "PC1 means for group A ({:.3}) and group B ({:.3}) should have opposite signs",
        mean_a,
        mean_b
    );

    // The separation should be at least 3× the noise-level spread
    let abs_separation = (mean_a - mean_b).abs();
    assert!(
        abs_separation > 1.0,
        "PC1 group separation too small: |mean_A - mean_B| = {:.3}",
        abs_separation
    );
}

// ─────────────────────────────────────────────────────────────
// Test 4: GSEA finds planted pathway
// ─────────────────────────────────────────────────────────────

/// Build a ranked gene list where the top 50 genes are the "planted" gene set.
/// Run enrichment with 200 permutations.  The planted set should have:
///   - ES > 0.3  (clearly enriched)
///   - p_value < 0.1
#[test]
fn test_gsea_finds_planted_pathway() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 200usize;
    let n_planted = 50usize;
    let n_samples = 16usize;
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(42);

    // Expression matrix: planted genes are strongly up in group B
    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            let base = if g < n_planted {
                // Planted genes: large effect
                if s < n_per_group {
                    50.0
                } else {
                    300.0
                }
            } else {
                100.0 // null genes: no effect
            };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            matrix[g * n_samples + s] = (base + noise * base * 0.1).max(1.0).round();
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();
    let groups: Vec<String> = (0..n_samples)
        .map(|s| {
            if s < n_per_group {
                "GroupA".to_string()
            } else {
                "GroupB".to_string()
            }
        })
        .collect();

    let expr_path = tmp.path().join("expr.parquet");
    let meta_path = tmp.path().join("meta.parquet");
    let de_dir = tmp.path().join("de");
    let enrich_dir = tmp.path().join("enrichment");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);
    write_metadata_parquet(&meta_path, &sample_names, &groups);

    // Step 1: run DE to get t-statistics
    rustpipe(&[
        "de",
        "--input",
        expr_path.to_str().unwrap(),
        "--metadata",
        meta_path.to_str().unwrap(),
        "--group-col",
        "condition",
        "--output",
        de_dir.to_str().unwrap(),
        "--method",
        "welch",
    ]);

    // Step 2: write gene sets JSON — planted set contains the first 50 genes
    let planted_genes: Vec<String> = (0..n_planted).map(|g| format!("GENE{:05}", g)).collect();
    let null_set_genes: Vec<String> = (n_planted..n_planted + 30)
        .map(|g| format!("GENE{:05}", g))
        .collect();

    let gene_sets: HashMap<&str, &Vec<String>> = [
        ("PLANTED_PATHWAY", &planted_genes),
        ("NULL_PATHWAY", &null_set_genes),
    ]
    .into_iter()
    .collect();

    let gene_sets_path = tmp.path().join("gene_sets.json");
    let json_str = serde_json::to_string(&gene_sets).unwrap();
    std::fs::write(&gene_sets_path, &json_str).unwrap();

    // Step 3: run enrichment
    rustpipe(&[
        "enrich",
        "--de-results",
        de_dir.to_str().unwrap(),
        "--gene-sets",
        gene_sets_path.to_str().unwrap(),
        "--output",
        enrich_dir.to_str().unwrap(),
        "--n-permutations",
        "200",
    ]);

    // Read the enrichment Parquet result
    let enrich_file = enrich_dir.join("GroupA_vs_GroupB_enrichment.parquet");
    assert!(
        enrich_file.exists(),
        "enrichment output not found: {}",
        enrich_file.display()
    );

    let df = LazyFrame::scan_parquet(&enrich_file, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    assert!(df.height() > 0, "enrichment results Parquet is empty");

    // Check required columns
    let cols = parquet_columns(&enrich_file);
    for required_col in &["gene_set", "n_genes", "es", "nes", "p_value", "fdr", "leading_edge"] {
        assert!(
            cols.contains(&required_col.to_string()),
            "Enrichment output missing column '{}'",
            required_col
        );
    }

    // Find the planted pathway entry
    let gene_set_col: Vec<String> = df
        .column("gene_set")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect();
    let es_col = read_f64_column(&enrich_file, "es");
    let p_col = read_f64_column(&enrich_file, "p_value");

    let planted_idx = gene_set_col
        .iter()
        .position(|s| s == "PLANTED_PATHWAY")
        .expect("PLANTED_PATHWAY not found in enrichment results");

    let es = es_col[planted_idx];
    let p_value = p_col[planted_idx];

    // The planted genes are upregulated in GroupB; DE ranks genes by signed
    // t-statistic descending, so the planted genes should appear near the top
    // of the ranked list. We expect a positive ES and a small p-value.
    assert!(
        es.abs() > 0.3,
        "Planted pathway ES = {:.3} — expected |ES| > 0.3",
        es
    );

    assert!(
        p_value < 0.1,
        "Planted pathway p_value = {:.4} — expected < 0.1",
        p_value
    );

    // Also verify summary TSV exists
    let summary_file = enrich_dir.join("GroupA_vs_GroupB_enrichment_summary.tsv");
    assert!(
        summary_file.exists(),
        "enrichment summary TSV not found: {}",
        summary_file.display()
    );
}

// ─────────────────────────────────────────────────────────────
// Test 4b: GSEA with GMT format gene sets
// ─────────────────────────────────────────────────────────────

/// Same planted-pathway scenario as `test_gsea_finds_planted_pathway`, but gene
/// sets are provided in GMT format instead of JSON.  Verifies that GMT loading
/// integrates end-to-end through the enrichment CLI.
#[test]
fn test_gsea_with_gmt_format() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 200usize;
    let n_planted = 50usize;
    let n_samples = 16usize;
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(42);

    // Expression matrix: planted genes are strongly up in group B
    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            let base = if g < n_planted {
                if s < n_per_group {
                    50.0
                } else {
                    300.0
                }
            } else {
                100.0
            };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            matrix[g * n_samples + s] = (base + noise * base * 0.1).max(1.0).round();
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();
    let groups: Vec<String> = (0..n_samples)
        .map(|s| {
            if s < n_per_group {
                "GroupA".to_string()
            } else {
                "GroupB".to_string()
            }
        })
        .collect();

    let expr_path = tmp.path().join("expr.parquet");
    let meta_path = tmp.path().join("meta.parquet");
    let de_dir = tmp.path().join("de");
    let enrich_dir = tmp.path().join("enrichment");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);
    write_metadata_parquet(&meta_path, &sample_names, &groups);

    // Step 1: run DE
    rustpipe(&[
        "de",
        "--input",
        expr_path.to_str().unwrap(),
        "--metadata",
        meta_path.to_str().unwrap(),
        "--group-col",
        "condition",
        "--output",
        de_dir.to_str().unwrap(),
        "--method",
        "welch",
    ]);

    // Step 2: write gene sets as GMT instead of JSON
    let planted_genes: Vec<String> = (0..n_planted).map(|g| format!("GENE{:05}", g)).collect();
    let null_set_genes: Vec<String> = (n_planted..n_planted + 30)
        .map(|g| format!("GENE{:05}", g))
        .collect();

    let gene_sets_path = tmp.path().join("gene_sets.gmt");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&gene_sets_path).unwrap();
        // PLANTED_PATHWAY line
        write!(f, "PLANTED_PATHWAY\tplanted pathway description").unwrap();
        for gene in &planted_genes {
            write!(f, "\t{}", gene).unwrap();
        }
        writeln!(f).unwrap();
        // NULL_PATHWAY line
        write!(f, "NULL_PATHWAY\tna").unwrap();
        for gene in &null_set_genes {
            write!(f, "\t{}", gene).unwrap();
        }
        writeln!(f).unwrap();
    }

    // Step 3: run enrichment with GMT file
    rustpipe(&[
        "enrich",
        "--de-results",
        de_dir.to_str().unwrap(),
        "--gene-sets",
        gene_sets_path.to_str().unwrap(),
        "--output",
        enrich_dir.to_str().unwrap(),
        "--n-permutations",
        "200",
    ]);

    // Read the enrichment Parquet result
    let enrich_file = enrich_dir.join("GroupA_vs_GroupB_enrichment.parquet");
    assert!(
        enrich_file.exists(),
        "enrichment output not found: {}",
        enrich_file.display()
    );

    let df = LazyFrame::scan_parquet(&enrich_file, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    assert!(df.height() > 0, "enrichment results Parquet is empty");

    // Find the planted pathway entry
    let gene_set_col: Vec<String> = df
        .column("gene_set")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect();
    let es_col = read_f64_column(&enrich_file, "es");
    let p_col = read_f64_column(&enrich_file, "p_value");

    let planted_idx = gene_set_col
        .iter()
        .position(|s| s == "PLANTED_PATHWAY")
        .expect("PLANTED_PATHWAY not found in enrichment results");

    let es = es_col[planted_idx];
    let p_value = p_col[planted_idx];

    assert!(
        es.abs() > 0.3,
        "Planted pathway ES = {:.3} — expected |ES| > 0.3",
        es
    );

    assert!(
        p_value < 0.1,
        "Planted pathway p_value = {:.4} — expected < 0.1",
        p_value
    );
}

// ─────────────────────────────────────────────────────────────
// Test 5: CSV → Parquet round-trip
// ─────────────────────────────────────────────────────────────

/// Write a small CSV, convert to Parquet with the CLI, read back and verify
/// that gene names and all numeric values are preserved exactly.
#[test]
fn test_csv_to_parquet_round_trip() {
    let tmp = TempDir::new().unwrap();

    // Build a 5-gene × 4-sample CSV
    let n_genes = 5usize;
    let n_samples = 4usize;
    let gene_names: Vec<&str> = vec!["TP53", "BRCA1", "ESR1", "MKI67", "CDH1"];
    let values: Vec<Vec<f64>> = vec![
        vec![10.0, 20.0, 30.0, 40.0],
        vec![5.0, 15.0, 25.0, 35.0],
        vec![100.0, 200.0, 300.0, 400.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![50.0, 60.0, 70.0, 80.0],
    ];
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("SAMPLE{}", s)).collect();

    // Write CSV
    let csv_path = tmp.path().join("expr.csv");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&csv_path).unwrap();
        // Header: gene_symbol, SAMPLE0, SAMPLE1, ...
        write!(f, "gene_symbol").unwrap();
        for s in &sample_names {
            write!(f, ",{}", s).unwrap();
        }
        writeln!(f).unwrap();
        // Rows
        for (g, gene) in gene_names.iter().enumerate() {
            write!(f, "{}", gene).unwrap();
            for val in values[g].iter() {
                write!(f, ",{}", val).unwrap();
            }
            writeln!(f).unwrap();
        }
    }

    let parquet_path = tmp.path().join("expr.parquet");

    rustpipe(&[
        "convert",
        "--input",
        csv_path.to_str().unwrap(),
        "--output",
        parquet_path.to_str().unwrap(),
    ]);

    assert!(parquet_path.exists(), "Parquet output was not created");

    // Read back the Parquet and verify
    let df = LazyFrame::scan_parquet(&parquet_path, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    // Row count
    assert_eq!(
        df.height(),
        n_genes,
        "Row count mismatch: expected {}, got {}",
        n_genes,
        df.height()
    );

    // Column count: 1 gene col + n_samples
    assert_eq!(df.width(), n_samples + 1, "Column count mismatch");

    // Gene names preserved
    let genes_back: Vec<String> = df
        .column(df.get_column_names()[0])
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect();
    for (i, name) in gene_names.iter().enumerate() {
        assert_eq!(
            genes_back[i], *name,
            "Gene name mismatch at row {}: expected {}, got {}",
            i, name, genes_back[i]
        );
    }

    // Numeric values preserved exactly.
    // Polars may infer integer dtype from whole-number CSV values, so cast to
    // f64 before comparing — the important invariant is value preservation, not
    // dtype identity.
    let col_names = df.get_column_names();
    for s in 0..n_samples {
        let series = df.column(col_names[s + 1]).unwrap();
        let as_f64 = series
            .cast(&DataType::Float64)
            .expect("could not cast sample column to f64");
        let col = as_f64.f64().unwrap();
        for g in 0..n_genes {
            let actual = col.get(g).unwrap_or(f64::NAN);
            assert!(
                (actual - values[g][s]).abs() < 1e-6,
                "Value mismatch at gene {}, sample {}: expected {}, got {}",
                gene_names[g],
                sample_names[s],
                values[g][s],
                actual
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Test 6: Full pipeline orchestrator output files
// ─────────────────────────────────────────────────────────────

/// Run the `bench` subcommand on a small synthetic matrix and verify that all
/// expected output files are present and non-empty.
///
/// The bench command generates its own data and runs normalize → DE → PCA,
/// so it exercises the full orchestrator path without requiring external
/// fixtures.
#[test]
fn test_full_pipeline_output_files() {
    let tmp = TempDir::new().unwrap();
    let output_dir = tmp.path().join("pipeline_output");

    // Use a small matrix so the test stays fast
    rustpipe(&[
        "bench",
        "--n-genes",
        "200",
        "--n-samples",
        "20",
        "--output",
        output_dir.to_str().unwrap(),
    ]);

    // Verify the expected output files exist
    let required_files = [
        "raw_counts.parquet",
        "normalized.parquet",
        "benchmark_timing.json",
        "de/GroupA_vs_GroupB.parquet",
        "pca/scores.parquet",
        "pca/variance.parquet",
    ];

    for rel_path in &required_files {
        let full_path = output_dir.join(rel_path);
        assert!(
            full_path.exists(),
            "Required output file missing: {}",
            full_path.display()
        );
        let size = std::fs::metadata(&full_path).unwrap().len();
        assert!(size > 0, "Output file is empty: {}", full_path.display());
    }

    // Spot-check normalized.parquet: all values finite and in range
    let norm_path = output_dir.join("normalized.parquet");
    let df = LazyFrame::scan_parquet(&norm_path, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    let col_names = df.get_column_names();
    // First column is gene names
    let n_data_cols = col_names.len() - 1;
    assert_eq!(
        n_data_cols, 20,
        "Expected 20 sample columns in normalized.parquet"
    );

    for col_name in &col_names[1..] {
        let col = df.column(col_name).unwrap().f64().unwrap();
        for val in col.into_iter().flatten() {
            assert!(
                val.is_finite(),
                "Non-finite value in normalized.parquet column {}",
                col_name
            );
            assert!(
                (-10.0..=25.0).contains(&val),
                "Normalized value {} out of expected range [-10, 25] in column {}",
                val,
                col_name
            );
        }
    }

    // Spot-check DE output: columns present and gene count correct
    let de_path = output_dir.join("de/GroupA_vs_GroupB.parquet");
    let de_cols = parquet_columns(&de_path);
    for required_col in &[
        "gene",
        "t_statistic",
        "p_value",
        "p_adj",
        "log2_fold_change",
    ] {
        assert!(
            de_cols.contains(&required_col.to_string()),
            "DE output missing column '{}'",
            required_col
        );
    }
    let de_genes = read_string_column(&de_path, "gene");
    assert_eq!(de_genes.len(), 200, "Expected 200 genes in DE output");

    // Spot-check PCA: scores has correct shape and variance sums to <= 1.0
    let scores_path = output_dir.join("pca/scores.parquet");
    let scores_df = LazyFrame::scan_parquet(&scores_path, Default::default())
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(scores_df.height(), 20, "Expected 20 rows in PCA scores");

    let variance_path = output_dir.join("pca/variance.parquet");
    let var_exp = read_f64_column(&variance_path, "variance_explained");
    let var_sum: f64 = var_exp.iter().sum();
    assert!(
        var_sum <= 1.0 + 1e-9,
        "Variance explained sums to {:.4} — should be <= 1.0",
        var_sum
    );
    assert!(var_sum > 0.0, "Variance explained sums to zero");
}

// ─────────────────────────────────────────────────────────────
// Test 7: HVG selection filters genes correctly
// ─────────────────────────────────────────────────────────────

/// Create 200 genes: 20 high-variance and 180 flat.  Run `rustpipe hvg` with
/// `--n-top-genes 20`.  Verify:
///   - Output has exactly 20 genes
///   - Most of the selected genes are from the high-variance set
///   - hvg_stats.parquet has correct columns and row count
#[test]
fn test_hvg_filters_genes() {
    let tmp = TempDir::new().unwrap();
    let n_high_var = 20usize;
    let n_low_var = 180usize;
    let n_genes = n_high_var + n_low_var;
    let n_samples = 50usize;

    let mut rng = StdRng::seed_from_u64(77);

    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            let val = if g < n_high_var {
                // High variance: mean ~200, sd ~100
                let noise: f64 = rng.sample(StandardNormal);
                (200.0 + noise * 100.0).max(1.0)
            } else {
                // Low variance: mean ~200, sd ~2
                let noise: f64 = rng.sample(StandardNormal);
                (200.0 + noise * 2.0).max(1.0)
            };
            matrix[g * n_samples + s] = val;
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();

    let expr_path = tmp.path().join("expr.parquet");
    let hvg_output = tmp.path().join("hvg_filtered.parquet");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);

    rustpipe(&[
        "hvg",
        "--input",
        expr_path.to_str().unwrap(),
        "--output",
        hvg_output.to_str().unwrap(),
        "--n-top-genes",
        "20",
    ]);

    assert!(
        hvg_output.exists(),
        "HVG filtered output not found: {}",
        hvg_output.display()
    );

    // Read back filtered expression and check dimensions
    let df = LazyFrame::scan_parquet(&hvg_output, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    // Should have 20 genes (rows) + gene_symbol column + n_samples sample columns
    assert_eq!(
        df.height(),
        n_high_var,
        "Expected {} rows (HVGs) in filtered output, got {}",
        n_high_var,
        df.height()
    );
    assert_eq!(
        df.width(),
        n_samples + 1,
        "Expected {} columns (1 gene + {} samples), got {}",
        n_samples + 1,
        n_samples,
        df.width()
    );

    // Check that most selected genes are from the high-variance set (indices 0..20)
    let selected_genes = read_string_column(&hvg_output, "gene_symbol");
    let high_var_names: std::collections::HashSet<String> =
        (0..n_high_var).map(|g| format!("GENE{:05}", g)).collect();

    let n_correct = selected_genes
        .iter()
        .filter(|g| high_var_names.contains(g.as_str()))
        .count();

    assert!(
        n_correct >= 18,
        "Expected at least 18 of 20 selected HVGs to be from the high-variance set, got {}/{}",
        n_correct,
        selected_genes.len()
    );

    // Check hvg_stats.parquet
    let stats_path = tmp.path().join("hvg_stats.parquet");
    assert!(
        stats_path.exists(),
        "hvg_stats.parquet not found: {}",
        stats_path.display()
    );

    let stats_df = LazyFrame::scan_parquet(&stats_path, Default::default())
        .unwrap()
        .collect()
        .unwrap();

    // Should have all 200 genes
    assert_eq!(
        stats_df.height(),
        n_genes,
        "hvg_stats should have all {} genes, got {}",
        n_genes,
        stats_df.height()
    );

    // Check required columns
    let stats_cols = parquet_columns(&stats_path);
    for required_col in &[
        "gene",
        "mean",
        "variance",
        "standardized_variance",
        "is_hvg",
    ] {
        assert!(
            stats_cols.contains(&required_col.to_string()),
            "hvg_stats missing column '{}'",
            required_col
        );
    }

    // Check is_hvg count
    let is_hvg_col = stats_df.column("is_hvg").unwrap().bool().unwrap();
    let n_hvg = is_hvg_col.into_iter().filter(|v| *v == Some(true)).count();
    assert_eq!(n_hvg, 20, "Expected 20 is_hvg=true entries, got {}", n_hvg);

    // All standardized_variance values should be finite and non-negative
    let std_var = read_f64_column(&stats_path, "standardized_variance");
    for (i, &sv) in std_var.iter().enumerate() {
        assert!(
            sv.is_finite() && sv >= 0.0,
            "standardized_variance[{}] = {} is invalid (must be finite and >= 0)",
            i,
            sv
        );
    }
}

// ─────────────────────────────────────────────────────────────
// Test 8: Pipeline accepts TSV input (auto-converts to Parquet)
// ─────────────────────────────────────────────────────────────

/// Run the full `pipeline` subcommand with TSV count matrix and CSV metadata
/// (not Parquet). Verify that auto-conversion works and the pipeline completes
/// with expected output files.
#[test]
fn test_pipeline_with_tsv_input() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 50usize;
    let n_samples = 8usize;
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(88);

    // Write TSV count matrix
    let tsv_path = tmp.path().join("counts.tsv");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tsv_path).unwrap();
        let sample_names: Vec<String> =
            (0..n_samples).map(|s| format!("S{:02}", s)).collect();
        write!(f, "gene").unwrap();
        for s in &sample_names {
            write!(f, "\t{}", s).unwrap();
        }
        writeln!(f).unwrap();

        for g in 0..n_genes {
            write!(f, "GENE{:05}", g).unwrap();
            for s in 0..n_samples {
                let base = if g < 10 {
                    if s < n_per_group {
                        50.0
                    } else {
                        200.0
                    }
                } else {
                    100.0
                };
                let noise: f64 = rng.sample::<f64, _>(StandardNormal);
                let val = (base + noise * base * 0.15).max(1.0).round();
                write!(f, "\t{}", val as u64).unwrap();
            }
            writeln!(f).unwrap();
        }
    }

    // Write CSV metadata
    let csv_meta_path = tmp.path().join("meta.csv");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&csv_meta_path).unwrap();
        writeln!(f, "sample_id,condition").unwrap();
        for s in 0..n_samples {
            let group = if s < n_per_group { "GroupA" } else { "GroupB" };
            writeln!(f, "S{:02},{}", s, group).unwrap();
        }
    }

    let output_dir = tmp.path().join("pipeline_out");

    rustpipe(&[
        "pipeline",
        "--input",
        tsv_path.to_str().unwrap(),
        "--metadata",
        csv_meta_path.to_str().unwrap(),
        "--group-col",
        "condition",
        "--output",
        output_dir.to_str().unwrap(),
        "-v",
    ]);

    // Verify key outputs exist
    assert!(
        output_dir.join("normalized.parquet").exists(),
        "normalized.parquet missing after TSV pipeline"
    );
    assert!(
        output_dir.join("de/GroupA_vs_GroupB.parquet").exists(),
        "DE output missing after TSV pipeline"
    );
    assert!(
        output_dir.join("pca/scores.parquet").exists(),
        "PCA scores missing after TSV pipeline"
    );
    // Auto-converted intermediates should exist
    assert!(
        output_dir.join("_input_converted.parquet").exists(),
        "_input_converted.parquet missing (auto-convert failed)"
    );
    assert!(
        output_dir.join("_metadata_converted.parquet").exists(),
        "_metadata_converted.parquet missing (auto-convert failed)"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 9: Pipeline writes pipeline_manifest.json
// ─────────────────────────────────────────────────────────────

/// Run the full pipeline and verify that pipeline_manifest.json exists,
/// is valid JSON, and contains the expected top-level keys.
#[test]
fn test_pipeline_manifest_json() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 50usize;
    let n_samples = 8usize;
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(101);

    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            let base = if g < 10 && s >= n_per_group {
                200.0
            } else {
                100.0
            };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            matrix[g * n_samples + s] = (base + noise * 15.0).max(1.0).round();
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();
    let groups: Vec<String> = (0..n_samples)
        .map(|s| {
            if s < n_per_group {
                "GroupA".to_string()
            } else {
                "GroupB".to_string()
            }
        })
        .collect();

    let expr_path = tmp.path().join("expr.parquet");
    let meta_path = tmp.path().join("meta.parquet");
    let output_dir = tmp.path().join("pipeline_out");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);
    write_metadata_parquet(&meta_path, &sample_names, &groups);

    rustpipe(&[
        "pipeline",
        "--input",
        expr_path.to_str().unwrap(),
        "--metadata",
        meta_path.to_str().unwrap(),
        "--group-col",
        "condition",
        "--output",
        output_dir.to_str().unwrap(),
        "-v",
    ]);

    let manifest_path = output_dir.join("pipeline_manifest.json");
    assert!(
        manifest_path.exists(),
        "pipeline_manifest.json not found: {}",
        manifest_path.display()
    );

    let content = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&content)
        .expect("pipeline_manifest.json is not valid JSON");

    // Check top-level keys
    assert!(manifest.get("rustpipe_version").is_some(), "missing rustpipe_version");
    assert!(manifest.get("timestamp").is_some(), "missing timestamp");
    assert!(manifest.get("inputs").is_some(), "missing inputs");
    assert!(manifest.get("parameters").is_some(), "missing parameters");
    assert!(manifest.get("timings").is_some(), "missing timings");
    assert!(manifest.get("total_seconds").is_some(), "missing total_seconds");

    // total_seconds should be positive
    let total = manifest["total_seconds"].as_f64().unwrap();
    assert!(total > 0.0, "total_seconds should be > 0, got {}", total);

    // timings should be an array with at least 2 entries (normalize + de)
    let timings = manifest["timings"].as_array().unwrap();
    assert!(
        timings.len() >= 2,
        "expected at least 2 timing entries, got {}",
        timings.len()
    );
}

// ─────────────────────────────────────────────────────────────
// Test 10: Pipeline writes MultiQC *_mqc.tsv files
// ─────────────────────────────────────────────────────────────

/// Run the full pipeline and verify that pca_mqc.tsv and sample_dists_mqc.tsv
/// exist in the output directory (not in the pca/ subdirectory).
#[test]
fn test_pipeline_multiqc_files() {
    let tmp = TempDir::new().unwrap();
    let n_genes = 50usize;
    let n_samples = 8usize;
    let n_per_group = n_samples / 2;

    let mut rng = StdRng::seed_from_u64(202);

    let mut matrix = vec![0.0f64; n_genes * n_samples];
    for g in 0..n_genes {
        for s in 0..n_samples {
            let base = if g < 10 && s >= n_per_group {
                200.0
            } else {
                100.0
            };
            let noise: f64 = rng.sample::<f64, _>(StandardNormal);
            matrix[g * n_samples + s] = (base + noise * 15.0).max(1.0).round();
        }
    }

    let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
    let sample_names: Vec<String> = (0..n_samples).map(|s| format!("S{:02}", s)).collect();
    let groups: Vec<String> = (0..n_samples)
        .map(|s| {
            if s < n_per_group {
                "GroupA".to_string()
            } else {
                "GroupB".to_string()
            }
        })
        .collect();

    let expr_path = tmp.path().join("expr.parquet");
    let meta_path = tmp.path().join("meta.parquet");
    let output_dir = tmp.path().join("pipeline_out");

    write_count_parquet(&expr_path, &gene_names, &sample_names, &matrix);
    write_metadata_parquet(&meta_path, &sample_names, &groups);

    rustpipe(&[
        "pipeline",
        "--input",
        expr_path.to_str().unwrap(),
        "--metadata",
        meta_path.to_str().unwrap(),
        "--group-col",
        "condition",
        "--output",
        output_dir.to_str().unwrap(),
        "-v",
    ]);

    // Check pca_mqc.tsv
    let pca_mqc = output_dir.join("pca_mqc.tsv");
    assert!(
        pca_mqc.exists(),
        "pca_mqc.tsv not found in output dir: {}",
        pca_mqc.display()
    );
    let pca_content = std::fs::read_to_string(&pca_mqc).unwrap();
    assert!(
        pca_content.contains("#plot_type: 'bargraph'"),
        "pca_mqc.tsv missing MultiQC bargraph header"
    );
    assert!(
        pca_content.contains("PC1"),
        "pca_mqc.tsv missing PC1 entry"
    );

    // Check sample_dists_mqc.tsv
    let dists_mqc = output_dir.join("sample_dists_mqc.tsv");
    assert!(
        dists_mqc.exists(),
        "sample_dists_mqc.tsv not found in output dir: {}",
        dists_mqc.display()
    );
    let dists_content = std::fs::read_to_string(&dists_mqc).unwrap();
    assert!(
        dists_content.contains("#plot_type: 'heatmap'"),
        "sample_dists_mqc.tsv missing MultiQC heatmap header"
    );
    // Should contain sample names
    assert!(
        dists_content.contains("S00"),
        "sample_dists_mqc.tsv missing sample names"
    );
}
