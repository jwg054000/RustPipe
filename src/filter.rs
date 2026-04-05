//! Low-count gene filtering matching edgeR's filterByExpr() defaults.
//!
//! Removes genes that do not meet a minimum expression threshold in a
//! minimum number of samples, reducing noise in downstream DE and
//! enrichment analyses.

use crate::io;
use anyhow::Result;
use log::info;
use std::path::Path;
use std::time::Instant;

/// Filter genes by minimum count threshold.
///
/// Keep a gene if at least `min_samples` samples have expression >= `min_count`.
///
/// # Arguments
/// * `data` — row-major matrix of shape `n_genes x n_samples`
/// * `n_genes` — number of genes (rows)
/// * `n_samples` — number of samples (columns)
/// * `gene_names` — gene identifiers, length `n_genes`
/// * `min_count` — minimum expression value a sample must have for the gene to count
/// * `min_samples` — minimum number of samples that must pass the threshold
///
/// # Returns
/// (filtered_gene_names, filtered_matrix, new_n_genes)
pub fn filter_low_counts(
    data: &[f64],
    n_genes: usize,
    n_samples: usize,
    gene_names: &[String],
    min_count: f64,
    min_samples: usize,
) -> (Vec<String>, Vec<f64>, usize) {
    let mut filtered_names = Vec::new();
    let mut filtered_data = Vec::new();

    for (g, name) in gene_names.iter().enumerate().take(n_genes) {
        let row_start = g * n_samples;
        let row = &data[row_start..row_start + n_samples];

        let passing_samples = row.iter().filter(|&&v| v >= min_count).count();

        if passing_samples >= min_samples {
            filtered_names.push(name.clone());
            filtered_data.extend_from_slice(row);
        }
    }

    let new_n_genes = filtered_names.len();
    (filtered_names, filtered_data, new_n_genes)
}

/// CLI entry point: read Parquet, filter low-count genes, write filtered Parquet.
pub fn run_filter(input: &Path, output: &Path, min_count: f64, min_samples: usize) -> Result<()> {
    let start = Instant::now();

    let (gene_names, sample_names, matrix, n_genes, n_samples) = io::load_expression_matrix(input)?;

    info!(
        "Filtering {}x{} matrix (min_count={}, min_samples={})",
        n_genes, n_samples, min_count, min_samples
    );

    let (filtered_names, filtered_data, new_n_genes) = filter_low_counts(
        &matrix,
        n_genes,
        n_samples,
        &gene_names,
        min_count,
        min_samples,
    );

    let removed = n_genes - new_n_genes;
    info!(
        "Kept {} genes, removed {} ({:.1}%) in {:.3}s",
        new_n_genes,
        removed,
        removed as f64 / n_genes as f64 * 100.0,
        start.elapsed().as_secs_f64()
    );

    io::write_expression_parquet(
        output,
        &filtered_names,
        &sample_names,
        &filtered_data,
        new_n_genes,
        n_samples,
        3,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gene_names(n: usize) -> Vec<String> {
        (0..n).map(|g| format!("GENE{:05}", g)).collect()
    }

    #[test]
    fn test_filter_removes_low_count_genes() {
        // 3 genes x 4 samples
        // Gene 0: [1, 2, 1, 2] — all below threshold of 10
        // Gene 1: [10, 20, 30, 40] — all pass
        // Gene 2: [0, 0, 0, 0] — all zero
        let data = vec![
            1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let gene_names = make_gene_names(3);

        let (filtered_names, filtered_data, new_n_genes) =
            filter_low_counts(&data, 3, 4, &gene_names, 10.0, 3);

        assert_eq!(new_n_genes, 1);
        assert_eq!(filtered_names, vec!["GENE00001"]);
        assert_eq!(filtered_data, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_filter_keeps_expressed_genes() {
        // 2 genes x 5 samples, both pass min_count=5, min_samples=3
        let data = vec![
            10.0, 20.0, 5.0, 15.0, 25.0, 100.0, 200.0, 300.0, 400.0, 500.0,
        ];
        let gene_names = make_gene_names(2);

        let (filtered_names, _filtered_data, new_n_genes) =
            filter_low_counts(&data, 2, 5, &gene_names, 5.0, 3);

        assert_eq!(new_n_genes, 2);
        assert_eq!(filtered_names.len(), 2);
    }

    #[test]
    fn test_filter_all_pass() {
        // All genes pass the threshold
        let data = vec![100.0, 200.0, 300.0, 150.0, 250.0, 350.0];
        let gene_names = make_gene_names(2);

        let (filtered_names, filtered_data, new_n_genes) =
            filter_low_counts(&data, 2, 3, &gene_names, 10.0, 2);

        assert_eq!(new_n_genes, 2);
        assert_eq!(filtered_names.len(), 2);
        assert_eq!(filtered_data.len(), 6);
    }

    #[test]
    fn test_filter_all_fail() {
        // No genes pass the threshold
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gene_names = make_gene_names(2);

        let (filtered_names, filtered_data, new_n_genes) =
            filter_low_counts(&data, 2, 3, &gene_names, 100.0, 2);

        assert_eq!(new_n_genes, 0);
        assert!(filtered_names.is_empty());
        assert!(filtered_data.is_empty());
    }
}
