//! Highly Variable Gene (HVG) selection using a variance-stabilizing
//! transformation (VST) approach.
//!
//! Implements the mean-variance relationship method (Seurat v3 / `vst`):
//! 1. For each gene, compute mean (mu) and variance (sigma^2) across cells.
//! 2. Fit expected_var = exp(a + b * ln(mean)) via OLS on log-log scale.
//! 3. Standardized variance = observed_var / expected_var.
//! 4. Rank by standardized variance, select top N.
//!
//! This is the recommended approach for single-cell workflows before PCA/DE.

use crate::io;
use anyhow::{bail, Result};
use log::info;
use polars::prelude::*;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Result of HVG selection.
pub struct HvgResult {
    /// Names of the selected highly variable genes.
    pub gene_names: Vec<String>,
    /// Row indices of the selected genes in the original matrix.
    pub gene_indices: Vec<usize>,
    /// Per-gene mean expression (all genes, not just selected).
    pub mean: Vec<f64>,
    /// Per-gene variance (all genes).
    pub variance: Vec<f64>,
    /// Per-gene standardized variance (all genes).
    pub standardized_variance: Vec<f64>,
}

/// Compute per-gene mean and variance from a row-major matrix.
///
/// Uses Welford's single-pass algorithm for numerical stability.
fn compute_gene_stats(data: &[f64], n_genes: usize, n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    let stats: Vec<(f64, f64)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let row_start = g * n_samples;
            let row = &data[row_start..row_start + n_samples];
            crate::stats::welford_mean_var(row)
        })
        .collect();

    let means: Vec<f64> = stats.iter().map(|(m, _)| *m).collect();
    let variances: Vec<f64> = stats.iter().map(|(_, v)| *v).collect();
    (means, variances)
}

/// Fit log(variance) ~ a + b * log(mean) via ordinary least squares.
///
/// Only uses genes with positive mean and positive variance for the fit.
/// Returns (intercept, slope).
fn fit_mean_variance_trend(means: &[f64], variances: &[f64]) -> (f64, f64) {
    // Collect (log_mean, log_var) for genes with positive mean and variance
    let pairs: Vec<(f64, f64)> = means
        .iter()
        .zip(variances.iter())
        .filter(|(&m, &v)| m > 0.0 && v > 0.0 && m.is_finite() && v.is_finite())
        .map(|(&m, &v)| (m.ln(), v.ln()))
        .collect();

    if pairs.len() < 2 {
        // Fallback: no trend, standardized variance = raw variance
        return (0.0, 1.0);
    }

    let n = pairs.len() as f64;
    let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = pairs.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        // Degenerate case: all means are identical
        let avg_log_var = sum_y / n;
        return (avg_log_var, 0.0);
    }

    let b = (n * sum_xy - sum_x * sum_y) / denom;
    let a = (sum_y - b * sum_x) / n;

    (a, b)
}

/// Select top `n_top_genes` highly variable genes.
///
/// Uses the VST approach:
/// 1. Compute per-gene mean and variance
/// 2. Fit expected variance as a function of mean (log-log linear regression)
/// 3. Standardized variance = observed / expected
/// 4. Rank genes by standardized variance, return top N
///
/// # Arguments
/// * `data` — row-major matrix of shape `n_genes x n_samples`
/// * `n_genes` — number of genes (rows)
/// * `n_samples` — number of samples (columns)
/// * `gene_names` — gene identifiers, length `n_genes`
/// * `n_top_genes` — how many HVGs to select
///
/// # Errors
/// Returns `PipeError::InvalidInput` if dimensions are inconsistent or
/// `n_top_genes` exceeds `n_genes`.
pub fn select_hvg(
    data: &[f64],
    n_genes: usize,
    n_samples: usize,
    gene_names: &[String],
    n_top_genes: usize,
) -> Result<HvgResult> {
    if data.len() != n_genes * n_samples {
        bail!(
            "Matrix size mismatch: expected {} elements, got {}",
            n_genes * n_samples,
            data.len()
        );
    }
    if gene_names.len() != n_genes {
        bail!(
            "Gene names length ({}) does not match n_genes ({})",
            gene_names.len(),
            n_genes
        );
    }
    if n_top_genes > n_genes {
        bail!(
            "n_top_genes ({}) exceeds total number of genes ({})",
            n_top_genes,
            n_genes
        );
    }
    if n_samples < 2 {
        bail!("Need at least 2 samples to compute variance");
    }

    let start = Instant::now();

    // Step 1: compute per-gene mean and variance
    let (means, variances) = compute_gene_stats(data, n_genes, n_samples);

    // Step 2: fit log(var) ~ a + b * log(mean)
    let (a, b) = fit_mean_variance_trend(&means, &variances);
    info!(
        "Mean-variance trend: ln(var) = {:.4} + {:.4} * ln(mean)",
        a, b
    );

    // Step 3: compute standardized variance for each gene
    let standardized_variance: Vec<f64> = means
        .par_iter()
        .zip(variances.par_iter())
        .map(|(&m, &v)| {
            if m <= 0.0 || !m.is_finite() || !v.is_finite() {
                // Genes with zero or negative mean get standardized variance of 0
                return 0.0;
            }
            let expected = (a + b * m.ln()).exp();
            if expected <= 0.0 || !expected.is_finite() {
                return 0.0;
            }
            let sv = v / expected;
            if sv.is_finite() {
                sv
            } else {
                0.0
            }
        })
        .collect();

    // Step 4: rank by standardized variance (descending), take top N
    let mut ranked_indices: Vec<usize> = (0..n_genes).collect();
    ranked_indices.sort_by(|&a_idx, &b_idx| {
        standardized_variance[b_idx]
            .partial_cmp(&standardized_variance[a_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked_indices.truncate(n_top_genes);

    // Sort the selected indices for stable ordering
    ranked_indices.sort();

    let selected_names: Vec<String> = ranked_indices
        .iter()
        .map(|&i| gene_names[i].clone())
        .collect();

    info!(
        "Selected {} HVGs from {} genes in {:.3}s",
        n_top_genes,
        n_genes,
        start.elapsed().as_secs_f64()
    );

    Ok(HvgResult {
        gene_names: selected_names,
        gene_indices: ranked_indices,
        mean: means,
        variance: variances,
        standardized_variance,
    })
}

/// Run HVG selection from the CLI: read input Parquet, select HVGs, write outputs.
///
/// Writes two files:
/// - `output`: filtered expression Parquet with only the top N genes
/// - `hvg_stats.parquet` (sibling of `output`): per-gene statistics
pub fn run_hvg(input: &Path, output: &Path, n_top_genes: usize) -> Result<()> {
    let start = Instant::now();

    let (gene_names, sample_names, matrix, n_genes, n_samples) = io::load_expression_matrix(input)?;

    info!(
        "Loaded {}x{} matrix, selecting top {} HVGs",
        n_genes, n_samples, n_top_genes
    );

    let result = select_hvg(&matrix, n_genes, n_samples, &gene_names, n_top_genes)?;

    // Build filtered matrix (only selected genes)
    let mut filtered = vec![0.0f64; n_top_genes * n_samples];
    for (new_g, &orig_g) in result.gene_indices.iter().enumerate() {
        let src_start = orig_g * n_samples;
        let dst_start = new_g * n_samples;
        filtered[dst_start..dst_start + n_samples]
            .copy_from_slice(&matrix[src_start..src_start + n_samples]);
    }

    // Write filtered expression
    io::write_expression_parquet(
        output,
        &result.gene_names,
        &sample_names,
        &filtered,
        n_top_genes,
        n_samples,
        3,
    )?;

    // Write HVG stats Parquet as a sibling of the output file
    let stats_path = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("hvg_stats.parquet");
    write_hvg_stats(
        &stats_path,
        &gene_names,
        &result.mean,
        &result.variance,
        &result.standardized_variance,
        &result.gene_indices,
    )?;

    info!(
        "HVG selection complete in {:.3}s",
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Write per-gene HVG statistics to a Parquet file.
///
/// Columns: gene, mean, variance, standardized_variance, is_hvg
fn write_hvg_stats(
    path: &Path,
    gene_names: &[String],
    means: &[f64],
    variances: &[f64],
    std_variances: &[f64],
    selected_indices: &[usize],
) -> Result<()> {
    use std::collections::HashSet;

    let selected_set: HashSet<usize> = selected_indices.iter().copied().collect();
    let is_hvg: Vec<bool> = (0..gene_names.len())
        .map(|i| selected_set.contains(&i))
        .collect();

    let df = DataFrame::new(vec![
        Series::new("gene".into(), gene_names),
        Series::new("mean".into(), means),
        Series::new("variance".into(), variances),
        Series::new("standardized_variance".into(), std_variances),
        Series::new("is_hvg".into(), &is_hvg),
    ])?;

    let file = std::fs::File::create(path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(Some(
            ZstdLevel::try_new(3).unwrap(),
        )))
        .finish(&mut df.clone())?;

    info!("Wrote HVG stats to {}", path.display());

    Ok(())
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    /// Helper: build a matrix with `n_high_var` high-variance genes and
    /// `n_low_var` low-variance genes.  Returns (matrix, gene_names, n_genes, n_samples).
    fn make_test_matrix(
        n_high_var: usize,
        n_low_var: usize,
        n_samples: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<String>, usize, usize) {
        let n_genes = n_high_var + n_low_var;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0f64; n_genes * n_samples];

        for g in 0..n_genes {
            for s in 0..n_samples {
                let val = if g < n_high_var {
                    // High variance: mean ~100, sd ~50
                    let noise: f64 = rng.sample(StandardNormal);
                    (100.0 + noise * 50.0).max(1.0)
                } else {
                    // Low variance: mean ~100, sd ~1
                    let noise: f64 = rng.sample(StandardNormal);
                    (100.0 + noise * 1.0).max(1.0)
                };
                matrix[g * n_samples + s] = val;
            }
        }

        let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
        (matrix, gene_names, n_genes, n_samples)
    }

    #[test]
    fn test_hvg_selects_variable_genes() {
        let n_high = 10;
        let n_low = 90;
        let n_samples = 50;
        let (matrix, gene_names, n_genes, _) = make_test_matrix(n_high, n_low, n_samples, 42);

        let result = select_hvg(&matrix, n_genes, n_samples, &gene_names, 10).unwrap();

        // All 10 selected genes should be from the high-variance set (indices 0..10)
        for &idx in &result.gene_indices {
            assert!(
                idx < n_high,
                "Selected gene index {} is not a high-variance gene (expected < {})",
                idx,
                n_high
            );
        }
    }

    #[test]
    fn test_hvg_top_n_respected() {
        let (matrix, gene_names, n_genes, n_samples) = make_test_matrix(20, 80, 30, 99);

        for n_top in [1, 5, 10, 20, 50] {
            let result = select_hvg(&matrix, n_genes, n_samples, &gene_names, n_top).unwrap();
            assert_eq!(
                result.gene_names.len(),
                n_top,
                "Expected {} HVGs, got {}",
                n_top,
                result.gene_names.len()
            );
            assert_eq!(
                result.gene_indices.len(),
                n_top,
                "Expected {} gene indices, got {}",
                n_top,
                result.gene_indices.len()
            );
        }
    }

    #[test]
    fn test_hvg_standardized_variance_finite() {
        let (matrix, gene_names, n_genes, n_samples) = make_test_matrix(10, 90, 40, 7);

        let result = select_hvg(&matrix, n_genes, n_samples, &gene_names, 10).unwrap();

        // All standardized variance values should be finite and non-negative
        for (i, &sv) in result.standardized_variance.iter().enumerate() {
            assert!(
                sv.is_finite(),
                "standardized_variance[{}] = {} is not finite",
                i,
                sv
            );
            assert!(
                sv >= 0.0,
                "standardized_variance[{}] = {} is negative",
                i,
                sv
            );
        }

        // All mean and variance values should also be finite
        for (i, (&m, &v)) in result.mean.iter().zip(result.variance.iter()).enumerate() {
            assert!(m.is_finite(), "mean[{}] = {} is not finite", i, m);
            assert!(v.is_finite(), "variance[{}] = {} is not finite", i, v);
            assert!(v >= 0.0, "variance[{}] = {} is negative", i, v);
        }
    }

    #[test]
    fn test_hvg_error_on_too_many() {
        let (matrix, gene_names, n_genes, n_samples) = make_test_matrix(5, 5, 10, 1);
        let result = select_hvg(&matrix, n_genes, n_samples, &gene_names, n_genes + 1);
        assert!(result.is_err(), "Should error when n_top_genes > n_genes");
    }

    #[test]
    fn test_hvg_error_on_single_sample() {
        let matrix = vec![1.0, 2.0, 3.0]; // 3 genes x 1 sample
        let gene_names = vec!["A".into(), "B".into(), "C".into()];
        let result = select_hvg(&matrix, 3, 1, &gene_names, 2);
        assert!(result.is_err(), "Should error when n_samples < 2");
    }

    #[test]
    fn test_fit_mean_variance_trend() {
        // Perfect power-law: var = mean^2 → log(var) = 2*log(mean)
        let means = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let variances: Vec<f64> = means.iter().map(|m| m * m).collect();

        let (a, b) = fit_mean_variance_trend(&means, &variances);
        // Should find a ≈ 0, b ≈ 2
        assert!((a).abs() < 0.1, "Intercept should be ~0, got {}", a);
        assert!((b - 2.0).abs() < 0.1, "Slope should be ~2.0, got {}", b);
    }
}
