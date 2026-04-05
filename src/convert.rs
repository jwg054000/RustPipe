//! CSV/TSV → Parquet conversion with Polars streaming.
//!
//! Handles the common bioinformatics CSV/TSV layout: first column is gene names,
//! remaining columns are numeric expression values (one per sample).
//! Uses Polars lazy + streaming to handle files larger than RAM.
//!
//! Supports:
//! - Automatic delimiter detection (tab vs comma)
//! - Gzip-compressed input files (.gz)

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use log::info;
use polars::prelude::*;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

/// Auto-detect the separator by counting tabs vs commas in the first line.
///
/// Returns `b'\t'` if tabs outnumber commas, `b','` otherwise.
fn detect_separator(path: &Path) -> Result<u8> {
    let first_line = {
        let content = std::fs::read_to_string(path).with_context(|| {
            format!(
                "Cannot read file for delimiter detection: {}",
                path.display()
            )
        })?;
        content.lines().next().unwrap_or("").to_string()
    };

    let n_tabs = first_line.matches('\t').count();
    let n_commas = first_line.matches(',').count();

    if n_tabs > n_commas {
        info!(
            "Auto-detected separator: TAB ({} tabs vs {} commas)",
            n_tabs, n_commas
        );
        Ok(b'\t')
    } else {
        info!(
            "Auto-detected separator: COMMA ({} commas vs {} tabs)",
            n_commas, n_tabs
        );
        Ok(b',')
    }
}

/// Decompress a .gz file to a temporary file and return its path.
///
/// The caller is responsible for the lifetime of the temp file (it lives in
/// the same directory as the input).
fn decompress_gz(gz_path: &Path) -> Result<std::path::PathBuf> {
    let gz_file = std::fs::File::open(gz_path)
        .with_context(|| format!("Cannot open gzip file: {}", gz_path.display()))?;
    let decoder = GzDecoder::new(BufReader::new(gz_file));

    let temp_path = gz_path.with_extension("tmp_decompressed");
    let tmp_file = std::fs::File::create(&temp_path).with_context(|| {
        format!(
            "Cannot create decompressed temp file: {}",
            temp_path.display()
        )
    })?;
    let mut writer = BufWriter::new(tmp_file);

    // Stream decompression: constant memory regardless of file size
    std::io::copy(&mut BufReader::new(decoder), &mut writer)
        .with_context(|| format!("Cannot decompress gzip file: {}", gz_path.display()))?;

    Ok(temp_path)
}

/// Convert a CSV/TSV expression matrix to Parquet format.
///
/// The CSV must have genes as rows and samples as columns, with the first
/// column containing gene identifiers (symbol or Ensembl ID).
///
/// Uses compression at the specified level (1-22).
///
/// If `separator` is `None`, auto-detects by inspecting the first line
/// (tab vs comma counts). Supports `.gz` compressed input.
pub fn csv_to_parquet(
    input: &Path,
    output: &Path,
    compression_level: u32,
    separator: Option<u8>,
) -> Result<()> {
    let start = Instant::now();

    // Handle .gz files: decompress to a temp file first
    let is_gz = input.extension().map(|e| e == "gz").unwrap_or(false);

    let (effective_input, _temp_file) = if is_gz {
        info!("Decompressing gzip input...");
        let temp = decompress_gz(input)?;
        (temp.clone(), Some(temp))
    } else {
        (input.to_path_buf(), None)
    };

    // Determine separator
    let sep = match separator {
        Some(s) => s,
        None => detect_separator(&effective_input)?,
    };

    info!("Scanning CSV schema from {}...", effective_input.display());

    // Read with Polars lazy for streaming
    let mut lf = LazyCsvReader::new(effective_input.to_string_lossy().to_string())
        .with_has_header(true)
        .with_separator(sep)
        .with_infer_schema_length(Some(1000))
        .finish()?;

    // Get schema info for logging
    let schema = lf.collect_schema().context("Cannot read CSV schema")?;
    let n_cols = schema.len();
    info!(
        "CSV has {} columns (1 gene + {} samples)",
        n_cols,
        n_cols - 1
    );

    // Collect and write to Parquet
    let mut df = lf.collect()?;

    let file = std::fs::File::create(output)?;
    let zstd_level =
        ZstdLevel::try_new(compression_level as i32).unwrap_or(ZstdLevel::try_new(3).unwrap());

    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(Some(zstd_level)))
        .with_row_group_size(Some(5000))
        .finish(&mut df)?;

    let out_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    let in_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);

    info!(
        "Converted {} → {} in {:.3}s ({:.1}MB → {:.1}MB, {:.1}x compression)",
        input.display(),
        output.display(),
        start.elapsed().as_secs_f64(),
        in_size as f64 / 1e6,
        out_size as f64 / 1e6,
        if out_size > 0 {
            in_size as f64 / out_size as f64
        } else {
            0.0
        },
    );

    // Clean up temp file if we created one
    if let Some(ref temp) = _temp_file {
        let _ = std::fs::remove_file(temp);
    }

    Ok(())
}
