//! TMM normalization and log2 CPM transform.
//!
//! Implements the trimmed mean of M-values (TMM) method from Robinson & Oshlack
//! (2010), matching edgeR's `calcNormFactors(method="TMM")` with default
//! parameters (30% M-value trimming, 5% A-value trimming).
//!
//! The pipeline: raw counts → TMM factors → effective library sizes → log2 CPM.

use crate::io;
use crate::stats;
use anyhow::Result;
use log::info;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Default M-value trim fraction (each tail).
const M_TRIM: f64 = 0.30;

/// Default A-value trim fraction (each tail).
const A_TRIM: f64 = 0.05;

/// Compute library sizes (column sums).
fn library_sizes(matrix: &[f64], n_genes: usize, n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .into_par_iter()
        .map(|s| {
            let mut sum = 0.0;
            for g in 0..n_genes {
                sum += matrix[g * n_samples + s];
            }
            sum
        })
        .collect()
}

/// Choose the reference sample (closest to upper-quartile mean).
///
/// Matches edgeR's default: the sample whose upper-quartile of nonzero counts
/// is closest to the geometric mean of all upper quartiles.
fn choose_reference(matrix: &[f64], n_genes: usize, n_samples: usize) -> usize {
    let uqs: Vec<f64> = (0..n_samples)
        .map(|s| {
            let mut nonzero: Vec<f64> = (0..n_genes)
                .map(|g| matrix[g * n_samples + s])
                .filter(|&v| v > 0.0)
                .collect();
            if nonzero.is_empty() {
                return 0.0;
            }
            nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q75_idx = (nonzero.len() as f64 * 0.75).ceil() as usize;
            nonzero[q75_idx.min(nonzero.len() - 1)]
        })
        .collect();

    // Geometric mean of upper quartiles
    let log_mean = uqs
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|v| v.ln())
        .sum::<f64>()
        / uqs.iter().filter(|&&v| v > 0.0).count().max(1) as f64;

    // Pick sample closest to the geometric mean
    uqs.iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .min_by(|(_, a), (_, b)| {
            (a.ln() - log_mean)
                .abs()
                .partial_cmp(&(b.ln() - log_mean).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute a single TMM factor for sample `s` vs reference `r`.
///
/// M = log2(count_s / lib_s) - log2(count_r / lib_r)
/// A = 0.5 * (log2(count_s / lib_s) + log2(count_r / lib_r))
///
/// Trim the outer `m_trim` fraction of M and `a_trim` fraction of A,
/// then compute the weighted mean of M (weights = inverse asymptotic variance).
#[allow(clippy::too_many_arguments)]
fn tmm_factor(
    matrix: &[f64],
    n_samples: usize,
    n_genes: usize,
    s: usize,
    r: usize,
    lib_s: f64,
    lib_r: f64,
    m_trim: f64,
    a_trim: f64,
) -> f64 {
    // Collect per-gene M and A values (only genes with positive counts in both)
    let mut ma: Vec<(f64, f64, f64)> = Vec::with_capacity(n_genes); // (M, A, weight)

    for g in 0..n_genes {
        let y_s = matrix[g * n_samples + s];
        let y_r = matrix[g * n_samples + r];

        if y_s <= 0.0 || y_r <= 0.0 {
            continue;
        }

        let w_s = y_s / lib_s;
        let w_r = y_r / lib_r;

        let m = (w_s / w_r).log2();
        // A = 0.5 * (log2(y_s/lib_s) + log2(y_r/lib_r)) — Robinson & Oshlack 2010
        let a = 0.5 * (w_s * w_r).log2();

        // Asymptotic variance (edgeR formula)
        let var = (lib_s - y_s) / (lib_s * y_s) + (lib_r - y_r) / (lib_r * y_r);
        if var <= 0.0 || !var.is_finite() || !m.is_finite() || !a.is_finite() {
            continue;
        }

        ma.push((m, a, 1.0 / var));
    }

    if ma.is_empty() {
        return 1.0;
    }

    let n = ma.len();

    // Sort by M, trim outer m_trim fraction
    let mut by_m: Vec<usize> = (0..n).collect();
    by_m.sort_by(|&i, &j| {
        ma[i]
            .0
            .partial_cmp(&ma[j].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let m_lo = (n as f64 * m_trim).floor() as usize;
    let m_hi = n - m_lo;

    // Sort by A, trim outer a_trim fraction
    let mut by_a: Vec<usize> = (0..n).collect();
    by_a.sort_by(|&i, &j| {
        ma[i]
            .1
            .partial_cmp(&ma[j].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let a_lo = (n as f64 * a_trim).floor() as usize;
    let a_hi = n - a_lo;

    // Keep genes that survive BOTH trims
    let mut keep = vec![true; n];

    // Mark M-trimmed
    for &idx in &by_m[..m_lo] {
        keep[idx] = false;
    }
    for &idx in &by_m[m_hi..] {
        keep[idx] = false;
    }

    // Mark A-trimmed
    for &idx in &by_a[..a_lo] {
        keep[idx] = false;
    }
    for &idx in &by_a[a_hi..] {
        keep[idx] = false;
    }

    // Weighted mean of M
    let mut sum_wm = 0.0;
    let mut sum_w = 0.0;
    for (i, &kept) in keep.iter().enumerate() {
        if kept {
            sum_wm += ma[i].2 * ma[i].0;
            sum_w += ma[i].2;
        }
    }

    if sum_w <= 0.0 {
        return 1.0;
    }

    // TMM factor = 2^(weighted_mean_M)
    2.0_f64.powf(sum_wm / sum_w)
}

/// Compute TMM normalization factors for all samples.
///
/// Returns `(factors, lib_sizes)` — a Vec of factors (one per sample) that
/// multiply with library sizes to give effective library sizes, along with
/// the raw library sizes so callers don't need to recompute them.
pub fn compute_tmm_factors(matrix: &[f64], n_genes: usize, n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    let lib_sizes = library_sizes(matrix, n_genes, n_samples);
    let ref_idx = choose_reference(matrix, n_genes, n_samples);

    info!("TMM reference sample index: {}", ref_idx);

    let factors: Vec<f64> = (0..n_samples)
        .into_par_iter()
        .map(|s| {
            if s == ref_idx {
                return 1.0;
            }
            tmm_factor(
                matrix,
                n_samples,
                n_genes,
                s,
                ref_idx,
                lib_sizes[s],
                lib_sizes[ref_idx],
                M_TRIM,
                A_TRIM,
            )
        })
        .collect();

    // Normalize so geometric mean of factors = 1
    let log_mean = factors.iter().map(|f| f.ln()).sum::<f64>() / n_samples as f64;
    let scale = (-log_mean).exp();

    let factors = factors.iter().map(|f| f * scale).collect();
    (factors, lib_sizes)
}

/// Apply TMM normalization and log2 CPM transform.
///
/// Reads raw counts → computes TMM factors → writes log2 CPM to output.
pub fn tmm_normalize(input: &Path, output: &Path, prior_count: f64) -> Result<()> {
    let start = Instant::now();

    let (gene_names, sample_names, matrix, n_genes, n_samples) = io::load_expression_matrix(input)?;

    info!("Computing TMM factors for {} samples...", n_samples);
    let (factors, lib_sizes) = compute_tmm_factors(&matrix, n_genes, n_samples);

    // Effective library sizes = lib_size * TMM_factor
    let eff_lib: Vec<f64> = lib_sizes
        .iter()
        .zip(factors.iter())
        .map(|(l, f)| l * f)
        .collect();

    info!("Applying log2 CPM transform (prior={})...", prior_count);

    // Transform in parallel (gene-parallel for cache locality)
    let mut normalized = vec![0.0f64; n_genes * n_samples];
    normalized
        .par_chunks_mut(n_samples)
        .enumerate()
        .for_each(|(g, row)| {
            for s in 0..n_samples {
                let count = matrix[g * n_samples + s];
                row[s] = stats::log2_cpm(count, eff_lib[s], prior_count);
            }
        });

    io::write_expression_parquet(
        output,
        &gene_names,
        &sample_names,
        &normalized,
        n_genes,
        n_samples,
        3,
    )?;

    info!(
        "TMM + log2CPM done in {:.3}s",
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_sizes() {
        // 2 genes × 3 samples, row-major
        let matrix = vec![
            10.0, 20.0, 30.0, // gene 0
            40.0, 50.0, 60.0, // gene 1
        ];
        let sizes = library_sizes(&matrix, 2, 3);
        assert_eq!(sizes, vec![50.0, 70.0, 90.0]);
    }

    #[test]
    fn test_tmm_factors_identical_samples() {
        // All columns identical → factors should all be ~1.0
        let n_genes = 100;
        let n_samples = 4;
        let mut matrix = vec![0.0; n_genes * n_samples];
        for g in 0..n_genes {
            let val = (g + 1) as f64 * 10.0;
            for s in 0..n_samples {
                matrix[g * n_samples + s] = val;
            }
        }

        let (factors, _lib_sizes) = compute_tmm_factors(&matrix, n_genes, n_samples);
        for f in &factors {
            assert!(
                (*f - 1.0).abs() < 0.01,
                "Factor should be ~1.0 for identical samples, got {}",
                f
            );
        }
    }
}
