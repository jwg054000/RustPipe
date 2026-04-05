//! Differential expression analysis with rayon parallelism.
//!
//! Supports three test methods:
//! - **Welch's t-test**: for bulk RNA-seq (unequal variances)
//! - **Wilcoxon rank-sum**: for single-cell data (non-parametric)
//! - **Moderated t-test**: empirical Bayes shrinkage (limma-style, Smyth 2004)
//!
//! All genes are tested in parallel using rayon, with BH FDR correction
//! applied across all p-values.

use crate::io;
use crate::stats;
use anyhow::{bail, Result};
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Run differential expression for all pairwise contrasts in a group column.
///
/// For each pair of groups (A vs B where A < B alphabetically), runs the
/// selected test across all genes in parallel, applies BH correction, and
/// writes results to `output_dir/A_vs_B.parquet`.
pub fn run_de(
    input: &Path,
    metadata: &Path,
    group_col: &str,
    output_dir: &Path,
    method: &str,
) -> Result<()> {
    let start = Instant::now();

    // Validate method
    if method != "welch" && method != "wilcoxon" && method != "moderated" {
        bail!(
            "Unknown DE method '{}'. Use 'welch', 'wilcoxon', or 'moderated'.",
            method
        );
    }

    // Load metadata
    let (meta_ids, meta_labels) = io::load_metadata(metadata, group_col)?;

    // Build sample→group map
    let sample_group: HashMap<String, String> = meta_ids.into_iter().zip(meta_labels).collect();

    // Load expression (only samples present in metadata)
    // Sort sample names for deterministic ordering (HashMap iteration is random)
    let mut sample_names: Vec<String> = sample_group.keys().cloned().collect();
    sample_names.sort();
    let (gene_names, matrix, n_genes, n_samples) =
        io::load_expression_pruned(input, &sample_names)?;

    // Identify unique groups
    let mut groups: Vec<String> = sample_group.values().cloned().collect();
    groups.sort();
    groups.dedup();
    info!("Groups ({}): {:?}", groups.len(), groups);

    std::fs::create_dir_all(output_dir)?;

    // Detect whether data is log-transformed (log2 CPM has negative values)
    let is_logged = matrix.iter().any(|&v| v < 0.0);
    info!(
        "Data appears {}-transformed ({} genes x {} samples)",
        if is_logged { "log2" } else { "raw" },
        n_genes,
        n_samples
    );

    // Run all pairwise comparisons
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            let group_a = &groups[i];
            let group_b = &groups[j];

            info!("DE: {} vs {} (method={})", group_a, group_b, method);

            let contrast_start = Instant::now();

            // Get sample indices for each group
            let idx_a: Vec<usize> = sample_names
                .iter()
                .enumerate()
                .filter(|(_, name)| {
                    sample_group
                        .get(*name)
                        .map(|g| g == group_a)
                        .unwrap_or(false)
                })
                .map(|(i, _)| i)
                .collect();

            let idx_b: Vec<usize> = sample_names
                .iter()
                .enumerate()
                .filter(|(_, name)| {
                    sample_group
                        .get(*name)
                        .map(|g| g == group_b)
                        .unwrap_or(false)
                })
                .map(|(i, _)| i)
                .collect();

            info!(
                "  {} samples in '{}', {} in '{}'",
                idx_a.len(),
                group_a,
                idx_b.len(),
                group_b
            );

            let (t_stats, p_values, log2_fc) = if method == "moderated" {
                // ── Moderated t-test (empirical Bayes, Smyth 2004) ──
                info!(
                    "DE: {} vs {} (method=moderated, empirical Bayes)",
                    group_a, group_b
                );

                let n_a = idx_a.len();
                let n_b = idx_b.len();
                let df_residual = (n_a + n_b - 2) as f64;

                // Pass 1: Compute per-gene pooled variance (parallel)
                let gene_variances: Vec<f64> = (0..n_genes)
                    .into_par_iter()
                    .map(|g| {
                        let row_start = g * n_samples;
                        let vals_a: Vec<f64> =
                            idx_a.iter().map(|&s| matrix[row_start + s]).collect();
                        let vals_b: Vec<f64> =
                            idx_b.iter().map(|&s| matrix[row_start + s]).collect();
                        let mean_a_g = vals_a.iter().sum::<f64>() / n_a as f64;
                        let mean_b_g = vals_b.iter().sum::<f64>() / n_b as f64;
                        let var_a: f64 = vals_a.iter().map(|&x| (x - mean_a_g).powi(2)).sum::<f64>()
                            / (n_a as f64 - 1.0).max(1.0);
                        let var_b: f64 = vals_b.iter().map(|&x| (x - mean_b_g).powi(2)).sum::<f64>()
                            / (n_b as f64 - 1.0).max(1.0);
                        // Pooled variance (equal variance assumption for moderated t)
                        ((n_a as f64 - 1.0) * var_a + (n_b as f64 - 1.0) * var_b) / df_residual
                    })
                    .collect();

                // Fit empirical Bayes prior
                let (d0, s0_sq) = stats::fit_ebayes_prior(&gene_variances, df_residual);
                info!("  Empirical Bayes prior: d0={:.1}, s0_sq={:.4}", d0, s0_sq);

                // Shrink variances
                let moderated_vars =
                    stats::moderate_variances(&gene_variances, df_residual, d0, s0_sq);
                let df_total = d0 + df_residual;

                // Pass 2: Compute moderated t-statistics and p-values (parallel)
                let results: Vec<(f64, f64, f64)> = (0..n_genes)
                    .into_par_iter()
                    .map(|g| {
                        let row_start = g * n_samples;
                        let vals_a: Vec<f64> =
                            idx_a.iter().map(|&s| matrix[row_start + s]).collect();
                        let vals_b: Vec<f64> =
                            idx_b.iter().map(|&s| matrix[row_start + s]).collect();
                        let mean_a_g = vals_a.iter().sum::<f64>() / n_a as f64;
                        let mean_b_g = vals_b.iter().sum::<f64>() / n_b as f64;

                        let se = (moderated_vars[g] * (1.0 / n_a as f64 + 1.0 / n_b as f64)).sqrt();
                        let t_stat = if se > 0.0 {
                            (mean_b_g - mean_a_g) / se
                        } else {
                            0.0
                        };
                        let p_value =
                            2.0 * stats::students_t_cdf(-t_stat.abs(), df_total);
                        let lfc = stats::log2_fold_change(&vals_a, &vals_b, is_logged);

                        (t_stat, p_value, lfc)
                    })
                    .collect();

                let t_stats: Vec<f64> = results.iter().map(|r| r.0).collect();
                let p_values: Vec<f64> = results.iter().map(|r| r.1).collect();
                let log2_fc: Vec<f64> = results.iter().map(|r| r.2).collect();

                (t_stats, p_values, log2_fc)
            } else {
                // ── Welch / Wilcoxon (parallel per-gene tests) ──
                let results: Vec<(f64, f64, f64)> = (0..n_genes)
                    .into_par_iter()
                    .map(|g| {
                        let row_start = g * n_samples;
                        let vals_a: Vec<f64> =
                            idx_a.iter().map(|&s| matrix[row_start + s]).collect();
                        let vals_b: Vec<f64> =
                            idx_b.iter().map(|&s| matrix[row_start + s]).collect();

                        let (stat, _extra, p) = match method {
                            "welch" => stats::welch_t_test(&vals_a, &vals_b),
                            "wilcoxon" => stats::wilcoxon_rank_sum(&vals_a, &vals_b),
                            _ => unreachable!(),
                        };

                        let lfc = stats::log2_fold_change(&vals_a, &vals_b, is_logged);

                        (stat, p, lfc)
                    })
                    .collect();

                let t_stats: Vec<f64> = results.iter().map(|r| r.0).collect();
                let p_values: Vec<f64> = results.iter().map(|r| r.1).collect();
                let log2_fc: Vec<f64> = results.iter().map(|r| r.2).collect();

                (t_stats, p_values, log2_fc)
            };

            // BH correction
            let mut adj_p = p_values.clone();
            stats::bh_adjust(&mut adj_p);

            // Count significant
            let n_sig = adj_p.iter().filter(|&&p| p < 0.05).count();
            info!(
                "  {} significant genes (padj < 0.05) in {:.3}s",
                n_sig,
                contrast_start.elapsed().as_secs_f64()
            );

            // Write results
            let out_path = output_dir.join(format!("{}_vs_{}.parquet", group_a, group_b));
            io::write_de_results(
                &out_path,
                &gene_names,
                &t_stats,
                &p_values,
                &adj_p,
                &log2_fc,
            )?;
        }
    }

    info!("DE complete in {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
