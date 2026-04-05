//! Gene set enrichment analysis (rank-based).
//!
//! Implements preranked GSEA using the Kolmogorov-Smirnov running-sum statistic
//! from Subramanian et al. (2005), with gene-label permutation for significance
//! and BH FDR correction.
//!
//! NES normalization follows Subramanian (2005): separate normalization for
//! positive and negative ES against the permutation null.

use crate::stats;
use anyhow::{bail, Result};
use log::info;
use polars::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Result of enrichment analysis for one gene set.
#[derive(Debug, Serialize)]
pub struct EnrichmentResult {
    pub gene_set: String,
    pub es: f64,
    pub nes: f64,
    pub p_value: f64,
    pub fdr: f64,
    pub n_genes: usize,
    pub leading_edge: Vec<String>,
}

/// Load gene sets from a JSON file.
///
/// Format: `{"set_name": ["GENE1", "GENE2", ...], ...}`
pub fn load_gene_sets_json(path: &Path) -> Result<HashMap<String, Vec<String>>> {
    let content = std::fs::read_to_string(path)?;
    let sets: HashMap<String, Vec<String>> = serde_json::from_str(&content)?;
    Ok(sets)
}

/// Parse a GMT (Gene Matrix Transposed) file into a gene sets map.
///
/// GMT is a tab-separated text file where each line defines one gene set:
///   `SET_NAME\tDESCRIPTION\tGENE1\tGENE2\t...`
///
/// - First field: gene set name
/// - Second field: description (often a URL or "na"; discarded)
/// - Remaining fields: gene symbols (variable number per line)
///
/// Empty lines and lines with fewer than 3 fields (no genes) are skipped.
/// This is the standard format used by MSigDB, Enrichr, and most gene set
/// databases.
pub fn load_gene_sets_gmt(path: &Path) -> Result<HashMap<String, Vec<String>>> {
    let content = std::fs::read_to_string(path)?;
    parse_gmt(&content)
}

/// Parse GMT-formatted text into a gene sets map.
///
/// Separated from `load_gene_sets_gmt` for testability.
fn parse_gmt(content: &str) -> Result<HashMap<String, Vec<String>>> {
    let mut sets = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();

        // Need at least name + description + one gene
        if fields.len() < 3 {
            continue;
        }

        let name = fields[0].to_string();
        let genes: Vec<String> = fields[2..]
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        if !genes.is_empty() {
            sets.insert(name, genes);
        }
    }

    if sets.is_empty() {
        bail!("GMT file contains no valid gene sets (expected tab-separated: NAME\\tDESC\\tGENE1\\tGENE2\\t...)");
    }

    Ok(sets)
}

/// Load gene sets from a file, auto-detecting format by extension.
///
/// - `.json` → JSON format `{"set_name": ["GENE1", ...]}`
/// - `.gmt`  → GMT format (tab-separated: NAME, DESCRIPTION, GENE1, GENE2, ...)
/// - other   → try JSON first, fall back to GMT
pub fn load_gene_sets(path: &Path) -> Result<HashMap<String, Vec<String>>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "json" => load_gene_sets_json(path),
        "gmt" => load_gene_sets_gmt(path),
        _ => {
            // Unknown extension: try JSON first, fall back to GMT
            load_gene_sets_json(path).or_else(|_| load_gene_sets_gmt(path))
        }
    }
}

/// Compute enrichment score from hit positions (no gene names needed).
///
/// `ranked_stats`: the full ranked statistic vector.
/// `hit_positions`: sorted indices of genes in the set within the ranked list.
/// Returns the ES (max deviation of the running sum).
#[allow(clippy::needless_range_loop)]
fn enrichment_score_positional(ranked_stats: &[f64], hit_positions: &[usize]) -> f64 {
    let n = ranked_stats.len();
    let n_hit = hit_positions.len();
    let n_miss = n - n_hit;

    if n_hit == 0 || n_miss == 0 {
        return 0.0;
    }

    // Weighted sum of absolute statistics for hits
    let hit_sum: f64 = hit_positions.iter().map(|&i| ranked_stats[i].abs()).sum();

    if hit_sum < 1e-15 {
        return 0.0;
    }

    let miss_penalty = 1.0 / n_miss as f64;

    // Running sum — use sorted hit pointer for O(n) walk
    let mut running_sum = 0.0;
    let mut max_es = 0.0;
    let mut min_es = 0.0;
    let mut hit_ptr = 0;

    for i in 0..n {
        if hit_ptr < n_hit && hit_positions[hit_ptr] == i {
            running_sum += ranked_stats[i].abs() / hit_sum;
            hit_ptr += 1;
        } else {
            running_sum -= miss_penalty;
        }

        if running_sum > max_es {
            max_es = running_sum;
        }
        if running_sum < min_es {
            min_es = running_sum;
        }
    }

    if max_es.abs() > min_es.abs() {
        max_es
    } else {
        min_es
    }
}

/// Compute the enrichment score (ES) for a named gene set.
///
/// Uses the weighted KS running-sum statistic. Returns (ES, leading_edge_indices).
#[allow(clippy::needless_range_loop)]
fn enrichment_score(
    ranked_genes: &[String],
    ranked_stats: &[f64],
    gene_set: &[String],
) -> (f64, Vec<usize>) {
    let n = ranked_genes.len();
    let gene_set_lookup: std::collections::HashSet<&str> =
        gene_set.iter().map(|s| s.as_str()).collect();

    // Find hit positions (sorted since we iterate in order)
    let hit_positions: Vec<usize> = ranked_genes
        .iter()
        .enumerate()
        .filter(|(_, g)| gene_set_lookup.contains(g.as_str()))
        .map(|(i, _)| i)
        .collect();

    let n_hit = hit_positions.len();
    if n_hit == 0 {
        return (0.0, vec![]);
    }

    // Compute ES using the positional function
    let es = enrichment_score_positional(ranked_stats, &hit_positions);

    // Find edge index for leading edge computation
    let n_miss = n - n_hit;
    if n_miss == 0 {
        return (es, hit_positions);
    }

    let hit_sum: f64 = hit_positions.iter().map(|&i| ranked_stats[i].abs()).sum();
    if hit_sum < 1e-15 {
        return (0.0, vec![]);
    }

    let miss_penalty = 1.0 / n_miss as f64;
    let mut running_sum = 0.0;
    let mut max_es = 0.0;
    let mut min_es = 0.0;
    let mut max_idx = 0;
    let mut min_idx = 0;
    let mut hit_ptr = 0;

    for i in 0..n {
        if hit_ptr < n_hit && hit_positions[hit_ptr] == i {
            running_sum += ranked_stats[i].abs() / hit_sum;
            hit_ptr += 1;
        } else {
            running_sum -= miss_penalty;
        }

        if running_sum > max_es {
            max_es = running_sum;
            max_idx = i;
        }
        if running_sum < min_es {
            min_es = running_sum;
            min_idx = i;
        }
    }

    let edge_end = if max_es.abs() > min_es.abs() {
        max_idx
    } else {
        min_idx
    };

    // Build in_set lookup for leading edge
    let hit_set: std::collections::HashSet<usize> = hit_positions.iter().copied().collect();

    let leading_edge = if es > 0.0 {
        (0..=edge_end).filter(|i| hit_set.contains(i)).collect()
    } else {
        (edge_end..n).filter(|i| hit_set.contains(i)).collect()
    };

    (es, leading_edge)
}

/// Generate permutation null distribution for a gene set of given size.
///
/// Shuffles gene-label positions `n_permutations` times, computes ES each time.
/// Uses a deterministic seed derived from `base_seed + set_index` for reproducibility.
fn permutation_null(
    ranked_stats: &[f64],
    n_hit: usize,
    n_permutations: usize,
    seed: u64,
) -> Vec<f64> {
    let n = ranked_stats.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut null_es = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Fisher-Yates partial shuffle: only need first n_hit elements
        for i in 0..n_hit {
            let j = rng.gen_range(i..n);
            indices.swap(i, j);
        }

        // Take first n_hit indices as the permuted hit set, sort for positional ES
        let mut perm_hits: Vec<usize> = indices[..n_hit].to_vec();
        perm_hits.sort_unstable();

        null_es.push(enrichment_score_positional(ranked_stats, &perm_hits));
    }

    null_es
}

/// Run enrichment analysis on DE results from a directory.
///
/// For each DE result file, ranks genes by t-statistic and tests all gene sets
/// with gene-label permutation for p-values and NES normalization.
pub fn run_enrichment(
    de_dir: &Path,
    gene_sets_path: &Path,
    output_dir: &Path,
    n_permutations: usize,
    seed: u64,
) -> Result<()> {
    let start = Instant::now();

    let gene_sets = load_gene_sets(gene_sets_path)?;
    info!("Loaded {} gene sets", gene_sets.len());
    info!("Using {} permutations per gene set", n_permutations);

    // Find all DE result Parquet files
    let de_files: Vec<_> = std::fs::read_dir(de_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false)
        })
        .collect();

    if de_files.is_empty() {
        bail!("No Parquet files found in {}", de_dir.display());
    }

    std::fs::create_dir_all(output_dir)?;

    for entry in &de_files {
        let de_path = entry.path();
        let contrast_name = de_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        info!("Enrichment for contrast: {}", contrast_name);

        // Load DE results
        let df = LazyFrame::scan_parquet(&de_path, Default::default())?.collect()?;

        let genes: Vec<String> = df
            .column("gene")?
            .str()?
            .into_iter()
            .map(|s| s.unwrap_or("").to_string())
            .collect();

        let t_stats: Vec<f64> = df
            .column("t_statistic")?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        // Rank genes by signed t-statistic descending (most positive first, most
        // negative last) — matches GSEA standard practice (Subramanian et al. 2005,
        // fgsea). Absolute-value ranking was incorrect: it collapsed up- and
        // down-regulated signal to the same end of the list, making directional
        // enrichment results unreliable.
        let mut indices: Vec<usize> = (0..genes.len()).collect();
        indices.sort_by(|&a, &b| {
            t_stats[b]
                .partial_cmp(&t_stats[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let ranked_genes: Vec<String> = indices.iter().map(|&i| genes[i].clone()).collect();
        let ranked_stats: Vec<f64> = indices.iter().map(|&i| t_stats[i]).collect();

        // Test all gene sets in parallel with permutation p-values
        let gene_set_vec: Vec<(&String, &Vec<String>)> = gene_sets.iter().collect();

        let mut results: Vec<EnrichmentResult> = gene_set_vec
            .par_iter()
            .enumerate()
            .map(|(set_idx, (name, set))| {
                let (es, le_idx) = enrichment_score(&ranked_genes, &ranked_stats, set);

                let leading_edge: Vec<String> =
                    le_idx.iter().map(|&i| ranked_genes[i].clone()).collect();

                let n_overlap = ranked_genes.iter().filter(|g| set.contains(g)).count();

                // Gene-label permutation null (deterministic seed per gene set)
                let null_es = permutation_null(
                    &ranked_stats,
                    n_overlap,
                    n_permutations,
                    seed + set_idx as u64,
                );

                // NES: normalize by mean of same-sign null (Subramanian 2005)
                let pos_null: Vec<f64> = null_es.iter().filter(|&&v| v > 0.0).cloned().collect();
                let neg_null: Vec<f64> = null_es.iter().filter(|&&v| v < 0.0).cloned().collect();

                let nes = if es >= 0.0 {
                    let mean_pos = if pos_null.is_empty() {
                        1.0
                    } else {
                        pos_null.iter().sum::<f64>() / pos_null.len() as f64
                    };
                    if mean_pos > 1e-15 {
                        es / mean_pos
                    } else {
                        0.0
                    }
                } else {
                    let mean_neg = if neg_null.is_empty() {
                        1.0
                    } else {
                        neg_null.iter().map(|v| v.abs()).sum::<f64>() / neg_null.len() as f64
                    };
                    if mean_neg > 1e-15 {
                        -(es.abs() / mean_neg)
                    } else {
                        0.0
                    }
                };

                // P-value: fraction of same-sign null ES at least as extreme
                // +1 Phipson-Smyth correction to avoid p=0
                let p_value = if es >= 0.0 {
                    let n_extreme = pos_null.iter().filter(|&&v| v >= es).count();
                    (n_extreme as f64 + 1.0) / (pos_null.len() as f64 + 1.0)
                } else {
                    let n_extreme = neg_null.iter().filter(|&&v| v <= es).count();
                    (n_extreme as f64 + 1.0) / (neg_null.len() as f64 + 1.0)
                };

                EnrichmentResult {
                    gene_set: (*name).clone(),
                    es,
                    nes,
                    p_value,
                    fdr: 1.0, // filled after BH
                    n_genes: n_overlap,
                    leading_edge,
                }
            })
            .collect();

        // BH FDR correction
        let mut pvals: Vec<f64> = results.iter().map(|r| r.p_value).collect();
        stats::bh_adjust(&mut pvals);
        for (r, &fdr) in results.iter_mut().zip(pvals.iter()) {
            r.fdr = fdr;
        }

        // Sort by FDR
        results.sort_by(|a, b| {
            a.fdr
                .partial_cmp(&b.fdr)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_sig = results.iter().filter(|r| r.fdr < 0.25).count();
        info!("  {} significant sets (FDR < 0.25)", n_sig);

        // Write results as Parquet
        {
            let gene_sets_col: Vec<&str> = results.iter().map(|r| r.gene_set.as_str()).collect();
            let n_genes_col: Vec<u32> = results.iter().map(|r| r.n_genes as u32).collect();
            let es_col: Vec<f64> = results.iter().map(|r| r.es).collect();
            let nes_col: Vec<f64> = results.iter().map(|r| r.nes).collect();
            let pvals_col: Vec<f64> = results.iter().map(|r| r.p_value).collect();
            let fdrs_col: Vec<f64> = results.iter().map(|r| r.fdr).collect();
            let le_col: Vec<String> =
                results.iter().map(|r| r.leading_edge.join(";")).collect();

            let mut df = DataFrame::new(vec![
                Series::new("gene_set".into(), &gene_sets_col),
                Series::new("n_genes".into(), &n_genes_col),
                Series::new("es".into(), &es_col),
                Series::new("nes".into(), &nes_col),
                Series::new("p_value".into(), &pvals_col),
                Series::new("fdr".into(), &fdrs_col),
                Series::new("leading_edge".into(), &le_col),
            ])?;

            let parquet_path =
                output_dir.join(format!("{}_enrichment.parquet", contrast_name));
            let file = std::fs::File::create(&parquet_path)?;
            ParquetWriter::new(file).finish(&mut df)?;
            info!("  Wrote {}", parquet_path.display());
        }

        // Write human-readable summary TSV (no leading_edge for scannability)
        {
            let summary_path =
                output_dir.join(format!("{}_enrichment_summary.tsv", contrast_name));
            let mut content = String::new();
            content.push_str("gene_set\tn_genes\tes\tnes\tp_value\tfdr\n");
            for r in &results {
                content.push_str(&format!(
                    "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
                    r.gene_set, r.n_genes, r.es, r.nes, r.p_value, r.fdr
                ));
            }
            std::fs::write(&summary_path, content)?;
            info!("  Wrote {}", summary_path.display());
        }
    }

    info!(
        "Enrichment complete in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a simple 3-line GMT string and verify gene set names and gene lists.
    #[test]
    fn test_gmt_parse_basic() {
        let gmt = "\
HALLMARK_APOPTOSIS\thttp://example.com/apoptosis\tTP53\tBAX\tBCL2\tCASP3
HALLMARK_MYC_TARGETS\tna\tMYC\tCDK4\tCCND1
KEGG_CELL_CYCLE\thttp://example.com/cycle\tCDK1\tCDK2\tRB1\tE2F1\tCCNB1";

        let sets = parse_gmt(gmt).expect("parse_gmt should succeed");

        assert_eq!(sets.len(), 3, "expected 3 gene sets");

        let apoptosis = &sets["HALLMARK_APOPTOSIS"];
        assert_eq!(
            apoptosis,
            &vec![
                "TP53".to_string(),
                "BAX".to_string(),
                "BCL2".to_string(),
                "CASP3".to_string()
            ]
        );

        let myc = &sets["HALLMARK_MYC_TARGETS"];
        assert_eq!(
            myc,
            &vec!["MYC".to_string(), "CDK4".to_string(), "CCND1".to_string()]
        );

        let cycle = &sets["KEGG_CELL_CYCLE"];
        assert_eq!(cycle.len(), 5);
        assert_eq!(cycle[0], "CDK1");
        assert_eq!(cycle[4], "CCNB1");
    }

    /// Empty lines and lines with only name+description (no genes) are skipped
    /// gracefully.
    #[test]
    fn test_gmt_parse_empty_lines() {
        let gmt = "\

HALLMARK_APOPTOSIS\thttp://example.com\tTP53\tBAX

NO_GENES_SET\tdescription_only

HALLMARK_P53\tna\tTP53\tMDM2\tCDKN1A
\t\t
";

        let sets = parse_gmt(gmt).expect("parse_gmt should succeed");

        // Only the two sets with actual genes should be present
        assert_eq!(
            sets.len(),
            2,
            "expected 2 gene sets (skipping empty/no-gene lines)"
        );
        assert!(sets.contains_key("HALLMARK_APOPTOSIS"));
        assert!(sets.contains_key("HALLMARK_P53"));
        assert!(!sets.contains_key("NO_GENES_SET"));

        assert_eq!(sets["HALLMARK_APOPTOSIS"], vec!["TP53", "BAX"]);
        assert_eq!(sets["HALLMARK_P53"], vec!["TP53", "MDM2", "CDKN1A"]);
    }

    /// A GMT string with no valid gene sets should return an error.
    #[test]
    fn test_gmt_parse_all_empty_returns_error() {
        let gmt = "\n\nNO_GENES\tdesc\n\n";
        let result = parse_gmt(gmt);
        assert!(
            result.is_err(),
            "expected error for GMT with no valid gene sets"
        );
    }

    /// Auto-detection via file extension works for .gmt files.
    #[test]
    fn test_load_gene_sets_auto_detect_gmt() {
        let dir = tempfile::TempDir::new().unwrap();
        let gmt_path = dir.path().join("test.gmt");
        std::fs::write(&gmt_path, "MY_SET\tdesc\tGENE1\tGENE2\tGENE3\n").unwrap();

        let sets = load_gene_sets(&gmt_path).expect("should load .gmt by extension");
        assert_eq!(sets.len(), 1);
        assert_eq!(sets["MY_SET"], vec!["GENE1", "GENE2", "GENE3"]);
    }

    /// Auto-detection via file extension works for .json files.
    #[test]
    fn test_load_gene_sets_auto_detect_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let json_path = dir.path().join("test.json");
        std::fs::write(&json_path, r#"{"MY_SET": ["GENE1", "GENE2"]}"#).unwrap();

        let sets = load_gene_sets(&json_path).expect("should load .json by extension");
        assert_eq!(sets.len(), 1);
        assert_eq!(sets["MY_SET"], vec!["GENE1", "GENE2"]);
    }

    /// Unknown extension: tries JSON first, falls back to GMT.
    #[test]
    fn test_load_gene_sets_fallback_gmt() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "SET_A\tdesc\tGENE1\tGENE2\n").unwrap();

        let sets = load_gene_sets(&path).expect("should fall back to GMT for .txt");
        assert_eq!(sets.len(), 1);
        assert_eq!(sets["SET_A"], vec!["GENE1", "GENE2"]);
    }
}
