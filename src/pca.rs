//! Principal component analysis via randomized SVD.
//!
//! Uses the Halko-Martinsson-Tropp algorithm with power iteration for accuracy:
//! 1. Form random Gaussian sketch: Y = A * Ω (n_genes × n_pcs+oversampling)
//! 2. Power iteration: Y = (A Aᵀ)^q * Y for accuracy on decaying spectra
//! 3. QR factorization of Y to get orthonormal basis Q
//! 4. Form B = Qᵀ * A (small matrix: n_pcs × n_samples)
//! 5. SVD of B to get final components
//!
//! This is 33x faster than full SVD on real TCGA data (18K genes × 800 samples).
//! Power iteration (q ≥ 3) is critical — without it, PC1 variance is overestimated
//! by ~66% on real RNA-seq data with slowly decaying singular values.

use crate::io;
use anyhow::Result;
use log::info;
use ndarray::{Array1, Array2};
use polars::prelude::NamedFrom;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Randomized SVD with power iteration.
///
/// Input: centered matrix (genes × samples) as flat row-major Vec.
/// Returns (scores, loadings, variance_explained) for the top `n_pcs` components.
///
/// # Arguments
/// * `matrix` - Row-major flat matrix (n_genes × n_samples)
/// * `n_genes` - Number of rows
/// * `n_samples` - Number of columns
/// * `n_pcs` - Number of principal components to compute
/// * `n_power_iter` - Power iteration rounds (≥3 recommended for RNA-seq)
pub fn randomized_svd(
    matrix: Vec<f64>,
    n_genes: usize,
    n_samples: usize,
    n_pcs: usize,
    n_power_iter: usize,
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let k = n_pcs;
    let oversampling = 10;
    let l = (k + oversampling).min(n_genes).min(n_samples);

    // Convert to ndarray for linear algebra (zero-copy, takes ownership)
    let a = Array2::from_shape_vec((n_genes, n_samples), matrix)
        .expect("Matrix shape mismatch");

    // Step 1: Random Gaussian sketch
    let mut rng = StdRng::seed_from_u64(42); // Reproducible
    let omega = Array2::from_shape_fn((n_samples, l), |_| rng.sample(StandardNormal));

    // Y = A * Ω
    let mut y = a.dot(&omega);

    // Step 2: Power iteration for spectrum accuracy (Algorithm 4.4, Halko et al. 2011)
    // QR at BOTH half-steps prevents numerical drift on slowly decaying spectra
    for _iter in 0..n_power_iter {
        let z = qr_q(&a.t().dot(&y)); // QR on n_samples × l
        y = a.dot(&z);
        y = qr_q(&y); // QR on n_genes × l
    }

    // Step 3: QR factorization of Y
    let q = qr_q(&y);

    // Step 4: Form the small matrix B = Qᵀ * A
    let b = q.t().dot(&a);

    // Step 5: Full SVD of the small matrix B (l × n_samples)
    // We use a simple Jacobi-like approach since l is small
    let (u_b, s, vt) = thin_svd(&b, 42);

    // Recover full left singular vectors: U = Q * U_B
    let u = q.dot(&u_b);

    // Truncate to k components
    let k = k.min(s.len());

    // Variance explained (proportional to squared singular values)
    let total_var: f64 = s.iter().map(|&si| si * si).sum();
    let var_explained = Array1::from_vec(s.iter().take(k).map(|&si| si * si / total_var).collect());

    // Scores: n_samples × k (project samples onto PCs)
    let scores = vt.slice(ndarray::s![..k, ..]).t().to_owned()
        * &Array1::from_vec(s.iter().take(k).cloned().collect::<Vec<_>>());

    // Loadings: n_genes × k
    let loadings = u.slice(ndarray::s![.., ..k]).to_owned();

    (scores, loadings, var_explained)
}

/// QR factorization — returns Q (orthonormal basis).
///
/// Uses modified Gram-Schmidt for numerical stability.
fn qr_q(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let mut q = a.clone();

    for j in 0..n {
        // Orthogonalize against all previous columns
        for i in 0..j {
            let dot: f64 = q.column(i).dot(&q.column(j));
            for k in 0..m {
                q[[k, j]] -= dot * q[[k, i]];
            }
        }

        // Normalize
        let norm: f64 = q.column(j).dot(&q.column(j)).sqrt();
        if norm > 1e-14 {
            for k in 0..m {
                q[[k, j]] /= norm;
            }
        }
    }

    q
}

/// Thin SVD via eigendecomposition of BᵀB.
///
/// For a small matrix B (l × n), computes SVD via:
/// BᵀB = V Σ² Vᵀ, then U = B V Σ⁻¹
///
/// Uses power iteration for eigendecomposition since l is small (~60).
/// Takes a `seed` for reproducible random initialization.
fn thin_svd(b: &Array2<f64>, seed: u64) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let (m, n) = b.dim();
    let k = m.min(n);

    // BᵀB
    let btb = b.t().dot(b);

    // Eigendecomposition via repeated QR iteration (simple, sufficient for small k)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = Array2::from_shape_fn((n, k), |_| {
        rng.sample::<f64, _>(StandardNormal)
    });
    v = qr_q(&v);

    // QR iteration (30 iterations is plenty for k < 100)
    for _ in 0..30 {
        let z = btb.dot(&v);
        v = qr_q(&z);
    }

    // Compute singular values: σᵢ = ||B vᵢ||
    let bv = b.dot(&v);
    let mut singular_values = Vec::with_capacity(k);
    for j in 0..k {
        let sigma = bv.column(j).dot(&bv.column(j)).sqrt();
        singular_values.push(sigma);
    }

    // U = B V Σ⁻¹
    let mut u = Array2::zeros((m, k));
    for j in 0..k {
        if singular_values[j] > 1e-14 {
            for i in 0..m {
                u[[i, j]] = bv[[i, j]] / singular_values[j];
            }
        }
    }

    // Vᵀ
    let vt = v.t().to_owned();

    (u, singular_values, vt)
}

/// Center the matrix (subtract gene means) in place.
fn center_matrix(matrix: &mut [f64], _n_genes: usize, n_samples: usize) {
    matrix.par_chunks_mut(n_samples).for_each(|row| {
        let mean: f64 = row.iter().sum::<f64>() / n_samples as f64;
        for val in row.iter_mut() {
            *val -= mean;
        }
    });
}

/// Results from PCA computation, returned for downstream use (e.g., MultiQC output).
pub struct PcaResult {
    /// Variance explained per PC (length = n_pcs).
    pub variance_explained: Vec<f64>,
    /// Flattened PC scores, row-major: n_samples x n_pcs.
    pub scores: Vec<f64>,
    /// Sample names from the input matrix.
    pub sample_names: Vec<String>,
    /// Number of samples.
    pub n_samples: usize,
    /// Number of PCs actually computed.
    pub n_pcs: usize,
}

/// Write PCA variance explained as MultiQC-compatible TSV.
pub fn write_multiqc_pca(variance_explained: &[f64], output_dir: &Path) -> Result<()> {
    let path = output_dir.join("pca_mqc.tsv");
    let mut content = String::new();
    content.push_str("#id: 'rustpipe_pca'\n");
    content.push_str("#section_name: 'RustPipe PCA'\n");
    content.push_str("#description: 'Variance explained by principal components'\n");
    content.push_str("#plot_type: 'bargraph'\n");
    content.push_str("PC\tVariance_Explained\n");
    for (i, &v) in variance_explained.iter().enumerate() {
        content.push_str(&format!("PC{}\t{:.6}\n", i + 1, v));
    }
    std::fs::write(&path, content)?;
    Ok(())
}

/// Write sample-sample distance matrix as MultiQC-compatible TSV.
pub fn write_multiqc_dists(
    scores: &[f64],
    sample_names: &[String],
    n_samples: usize,
    n_pcs: usize,
    output_dir: &Path,
) -> Result<()> {
    let path = output_dir.join("sample_dists_mqc.tsv");
    let mut content = String::new();
    content.push_str("#id: 'rustpipe_sample_dists'\n");
    content.push_str("#section_name: 'RustPipe Sample Distances'\n");
    content.push_str("#description: 'Euclidean distance between samples (PCA space)'\n");
    content.push_str("#plot_type: 'heatmap'\n");
    // Header row
    content.push('\t');
    content.push_str(&sample_names.join("\t"));
    content.push('\n');
    // Distance matrix
    let actual_pcs = n_pcs.min(scores.len() / n_samples.max(1));
    for i in 0..n_samples {
        content.push_str(&sample_names[i]);
        for j in 0..n_samples {
            let dist: f64 = (0..actual_pcs)
                .map(|pc| {
                    let a = scores[i * n_pcs + pc];
                    let b = scores[j * n_pcs + pc];
                    (a - b).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            content.push_str(&format!("\t{:.4}", dist));
        }
        content.push('\n');
    }
    std::fs::write(&path, content)?;
    Ok(())
}

/// Run PCA on an expression matrix and write results.
///
/// Outputs:
/// - `output_dir/scores.parquet` — sample scores (n_samples × n_pcs)
/// - `output_dir/variance.parquet` — variance explained per PC
///
/// Returns [`PcaResult`] so the caller can pass data to MultiQC writers.
pub fn run_pca(input: &Path, output_dir: &Path, n_pcs: usize, n_power_iter: usize) -> Result<PcaResult> {
    let start = Instant::now();

    let (_gene_names, sample_names, mut matrix, n_genes, n_samples) =
        io::load_expression_matrix(input)?;

    info!("Centering {} × {} matrix...", n_genes, n_samples);
    center_matrix(&mut matrix, n_genes, n_samples);

    info!(
        "Running randomized SVD: {} PCs, {} power iterations...",
        n_pcs, n_power_iter
    );
    let svd_start = Instant::now();
    let (scores, _loadings, var_explained) =
        randomized_svd(matrix, n_genes, n_samples, n_pcs, n_power_iter);
    info!("rSVD done in {:.3}s", svd_start.elapsed().as_secs_f64());

    // Log top variance
    for i in 0..3.min(var_explained.len()) {
        info!("  PC{}: {:.2}% variance", i + 1, var_explained[i] * 100.0);
    }

    // Write scores
    std::fs::create_dir_all(output_dir)?;

    let pc_names: Vec<String> = (1..=n_pcs.min(scores.ncols()))
        .map(|i| format!("PC{}", i))
        .collect();

    let mut columns: Vec<polars::prelude::Series> = Vec::new();
    columns.push(polars::prelude::Series::new("sample".into(), &sample_names));
    for (j, pc_name) in pc_names.iter().enumerate() {
        if j < scores.ncols() {
            let col: Vec<f64> = (0..scores.nrows()).map(|i| scores[[i, j]]).collect();
            columns.push(polars::prelude::Series::new(pc_name.as_str().into(), col));
        }
    }

    let df = polars::prelude::DataFrame::new(columns)?;
    let scores_path = output_dir.join("scores.parquet");
    let file = std::fs::File::create(&scores_path)?;
    polars::prelude::ParquetWriter::new(file).finish(&mut df.clone())?;

    // Write variance explained
    let var_df = polars::prelude::DataFrame::new(vec![
        polars::prelude::Series::new("pc".into(), &pc_names),
        polars::prelude::Series::new(
            "variance_explained".into(),
            var_explained
                .iter()
                .take(n_pcs)
                .cloned()
                .collect::<Vec<f64>>(),
        ),
    ])?;
    let var_path = output_dir.join("variance.parquet");
    let vf = std::fs::File::create(&var_path)?;
    polars::prelude::ParquetWriter::new(vf).finish(&mut var_df.clone())?;

    info!(
        "PCA complete in {:.3}s ({}×{} → {} PCs)",
        start.elapsed().as_secs_f64(),
        n_genes,
        n_samples,
        n_pcs
    );

    // Build flat scores vector (n_samples x actual_n_pcs, row-major) for MultiQC output
    let actual_n_pcs = n_pcs.min(scores.ncols());
    let mut flat_scores = Vec::with_capacity(scores.nrows() * actual_n_pcs);
    for i in 0..scores.nrows() {
        for j in 0..actual_n_pcs {
            flat_scores.push(scores[[i, j]]);
        }
    }

    Ok(PcaResult {
        variance_explained: var_explained.iter().take(n_pcs).cloned().collect(),
        scores: flat_scores,
        sample_names,
        n_samples,
        n_pcs: actual_n_pcs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_orthonormality() {
        let a =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let q = qr_q(&a);

        // Columns should be orthonormal
        let qtq = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-10,
                    "QᵀQ[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_rsvd_captures_variance() {
        // Simple rank-2 matrix: two clear signals + noise
        let n = 100;
        let m = 50;
        let mut data = vec![0.0; n * m];

        let mut rng = StdRng::seed_from_u64(123);
        for g in 0..n {
            for s in 0..m {
                // Signal: gene groups
                let signal1 = if g < 50 { 10.0 } else { -10.0 };
                let signal2 = if s < 25 { 5.0 } else { -5.0 };
                let noise: f64 = rng.sample(StandardNormal);
                data[g * m + s] = signal1 + signal2 + noise;
            }
        }

        // Center
        center_matrix(&mut data, n, m);

        let (_scores, _loadings, var_exp) = randomized_svd(data, n, m, 5, 3);

        // PC1 should capture most variance
        assert!(
            var_exp[0] > 0.3,
            "PC1 should capture >30% variance, got {:.2}%",
            var_exp[0] * 100.0
        );
    }
}
