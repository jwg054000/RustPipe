//! I/O utilities for reading/writing expression matrices and metadata.
//!
//! Expression matrices are stored as Parquet in "genes × samples" layout:
//! - First column: gene names (string)
//! - Remaining columns: one per sample (f64 expression values)
//!
//! This matches the common CSV layout from R/Bioconductor.

use anyhow::{Context, Result};
use log::info;
use polars::prelude::NamedFrom;
use polars::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Load an expression matrix from Parquet.
///
/// Returns (gene_names, sample_names, matrix) where matrix is a flat Vec<f64>
/// in row-major order (genes × samples). All downstream operations share this
/// single allocation — no redundant reads.
#[allow(clippy::type_complexity)]
pub fn load_expression_matrix(
    path: &Path,
) -> Result<(Vec<String>, Vec<String>, Vec<f64>, usize, usize)> {
    let start = Instant::now();

    let df = LazyFrame::scan_parquet(path, Default::default())?.collect()?;

    let gene_col = df
        .column(df.get_column_names()[0])
        .context("Cannot read gene column")?;
    let gene_names: Vec<String> = gene_col
        .str()
        .map_err(|_| anyhow::anyhow!("First column must be string (gene names)"))?
        .into_iter()
        .map(|s| s.unwrap_or("").to_string())
        .collect();

    let sample_names: Vec<String> = df.get_column_names()[1..]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let n_genes = gene_names.len();
    let n_samples = sample_names.len();

    // Build flat f64 matrix directly in row-major order (genes × samples).
    // Scatter each column's values into the correct row-major positions in a single pass.
    // Columns may be i64 (from integer CSV/TSV) — cast to f64 for uniform handling.
    let mut row_major = vec![0.0f64; n_genes * n_samples];
    for (s, col_name) in sample_names.iter().enumerate() {
        let col = df
            .column(col_name.as_str())
            .map_err(|_| anyhow::anyhow!("column not found: {}", col_name))?;
        let as_f64 = col
            .cast(&DataType::Float64)
            .map_err(|_| anyhow::anyhow!("Column '{}' is not numeric", col_name))?;
        let values = as_f64
            .f64()
            .map_err(|_| anyhow::anyhow!("Column '{}' could not be cast to f64", col_name))?;

        for (g, val) in values.into_iter().enumerate() {
            row_major[g * n_samples + s] = val.unwrap_or(0.0);
        }
    }

    info!(
        "Loaded {}×{} matrix from {} in {:.3}s",
        n_genes,
        n_samples,
        path.display(),
        start.elapsed().as_secs_f64()
    );

    Ok((gene_names, sample_names, row_major, n_genes, n_samples))
}

/// Load expression matrix with column pruning (only specified samples).
///
/// Much faster than full load when you only need a subset of samples.
pub fn load_expression_pruned(
    path: &Path,
    sample_names: &[String],
) -> Result<(Vec<String>, Vec<f64>, usize, usize)> {
    let start = Instant::now();

    let gene_col_name = {
        let schema = LazyFrame::scan_parquet(path, Default::default())?
            .collect_schema()
            .context("Cannot read schema")?;
        let name = schema
            .iter_names()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty Parquet file"))?
            .to_string();
        name
    };

    let mut cols_to_load: Vec<String> = vec![gene_col_name.clone()];
    cols_to_load.extend(sample_names.iter().cloned());

    let df = LazyFrame::scan_parquet(path, Default::default())?
        .select(
            cols_to_load
                .iter()
                .map(|c| col(c.as_str()))
                .collect::<Vec<_>>(),
        )
        .collect()?;

    let gene_col = df.column(&gene_col_name)?;
    let gene_names: Vec<String> = gene_col
        .str()?
        .into_iter()
        .map(|s| s.unwrap_or("").to_string())
        .collect();

    let n_genes = gene_names.len();
    let n_samples = sample_names.len();

    let mut row_major = vec![0.0f64; n_genes * n_samples];
    for (s, col_name) in sample_names.iter().enumerate() {
        let col = df.column(col_name.as_str())?;
        let as_f64 = col.cast(&DataType::Float64)?;
        let values = as_f64.f64()?;
        for (g, val) in values.into_iter().enumerate() {
            row_major[g * n_samples + s] = val.unwrap_or(0.0);
        }
    }

    info!(
        "Pruned load: {}×{} from {} in {:.3}s",
        n_genes,
        n_samples,
        path.display(),
        start.elapsed().as_secs_f64()
    );

    Ok((gene_names, row_major, n_genes, n_samples))
}

/// Load metadata Parquet and extract a grouping column.
///
/// Returns (sample_ids, group_labels).
pub fn load_metadata(path: &Path, group_col: &str) -> Result<(Vec<String>, Vec<String>)> {
    let df = LazyFrame::scan_parquet(path, Default::default())?.collect()?;

    let id_col = df.column(df.get_column_names()[0])?;
    let ids: Vec<String> = id_col
        .str()
        .map_err(|_| anyhow::anyhow!("First metadata column must be string IDs"))?
        .into_iter()
        .map(|s| s.unwrap_or("").to_string())
        .collect();

    let group = df
        .column(group_col)
        .map_err(|_| anyhow::anyhow!("column not found: {}", group_col))?;
    let labels: Vec<String> = group
        .str()
        .map_err(|_| anyhow::anyhow!("'{}' column must be string", group_col))?
        .into_iter()
        .map(|s| s.unwrap_or("").to_string())
        .collect();

    Ok((ids, labels))
}

/// Write a flat matrix to Parquet with gene names as first column.
pub fn write_expression_parquet(
    path: &Path,
    gene_names: &[String],
    sample_names: &[String],
    matrix: &[f64], // row-major: genes × samples
    n_genes: usize,
    n_samples: usize,
    compression_level: u32,
) -> Result<()> {
    let start = Instant::now();

    let mut columns: Vec<Series> = Vec::with_capacity(n_samples + 1);
    columns.push(Series::new("gene_symbol".into(), gene_names));

    for s in 0..n_samples {
        let col_data: Vec<f64> = (0..n_genes).map(|g| matrix[g * n_samples + s]).collect();
        columns.push(Series::new(sample_names[s].as_str().into(), col_data));
    }

    let df = DataFrame::new(columns)?;

    let file = std::fs::File::create(path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(Some(
            ZstdLevel::try_new(compression_level as i32).unwrap_or(ZstdLevel::try_new(3).unwrap()),
        )))
        .with_row_group_size(Some(5000))
        .finish(&mut df.clone())?;

    info!(
        "Wrote {}×{} to {} in {:.3}s",
        n_genes,
        n_samples,
        path.display(),
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Write DE results to Parquet.
pub fn write_de_results(
    path: &Path,
    gene_names: &[String],
    t_stats: &[f64],
    p_values: &[f64],
    adj_p_values: &[f64],
    log2_fc: &[f64],
) -> Result<()> {
    let df = DataFrame::new(vec![
        Series::new("gene".into(), gene_names),
        Series::new("t_statistic".into(), t_stats),
        Series::new("p_value".into(), p_values),
        Series::new("p_adj".into(), adj_p_values),
        Series::new("log2_fold_change".into(), log2_fc),
    ])?;

    let file = std::fs::File::create(path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(Some(
            ZstdLevel::try_new(3).unwrap(),
        )))
        .finish(&mut df.clone())?;

    Ok(())
}
