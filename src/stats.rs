//! Statistical functions: Welch's t-test, Wilcoxon rank-sum, BH correction,
//! Welford's online mean/variance, log2 CPM.
//!
//! All functions are designed for vectorized use with rayon parallel iterators.

use statrs::distribution::{ContinuousCDF, StudentsT};

/// Welford's single-pass online mean and variance.
///
/// Returns (mean, variance) computed in one pass with numerical stability.
/// Variance uses Bessel's correction (n-1 denominator).
#[inline]
pub fn welford_mean_var(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    if n == 1 {
        return (data[0], 0.0);
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    (mean, m2 / (n - 1) as f64)
}

/// Welch's t-test for two independent samples with unequal variances.
///
/// Returns (t_statistic, degrees_of_freedom, two_sided_p_value).
/// Uses the Welch-Satterthwaite approximation for degrees of freedom.
pub fn welch_t_test(group1: &[f64], group2: &[f64]) -> (f64, f64, f64) {
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;

    if n1 < 2.0 || n2 < 2.0 {
        return (f64::NAN, f64::NAN, 1.0);
    }

    let (mean1, var1) = welford_mean_var(group1);
    let (mean2, var2) = welford_mean_var(group2);

    let se1 = var1 / n1;
    let se2 = var2 / n2;
    let se_sum = se1 + se2;

    if se_sum < 1e-15 {
        return (0.0, n1 + n2 - 2.0, 1.0);
    }

    let t = (mean1 - mean2) / se_sum.sqrt();

    // Welch-Satterthwaite degrees of freedom
    let df = (se_sum * se_sum) / (se1 * se1 / (n1 - 1.0) + se2 * se2 / (n2 - 1.0));

    // Two-sided p-value from Student's t distribution
    let p = if df > 0.0 && df.is_finite() && t.is_finite() {
        match StudentsT::new(0.0, 1.0, df) {
            Ok(dist) => 2.0 * (1.0 - dist.cdf(t.abs())),
            Err(_) => 1.0,
        }
    } else {
        1.0
    };

    (t, df, p)
}

/// Wilcoxon rank-sum test (Mann-Whitney U) for two independent samples.
///
/// Returns (u_statistic, z_score, two_sided_p_value).
/// Uses normal approximation with continuity correction for large samples.
pub fn wilcoxon_rank_sum(group1: &[f64], group2: &[f64]) -> (f64, f64, f64) {
    let n1 = group1.len();
    let n2 = group2.len();

    if n1 == 0 || n2 == 0 {
        return (f64::NAN, f64::NAN, 1.0);
    }

    // Combine and rank
    let mut combined: Vec<(f64, u8)> = Vec::with_capacity(n1 + n2);
    for &x in group1 {
        combined.push((x, 0)); // 0 = group1
    }
    for &x in group2 {
        combined.push((x, 1)); // 1 = group2
    }
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks with tie averaging
    let n = combined.len();
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (combined[j].0 - combined[i].0).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0; // 1-based average
        for rank in &mut ranks[i..j] {
            *rank = avg_rank;
        }
        i = j;
    }

    // Sum ranks for group1
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|(c, _)| c.1 == 0)
        .map(|(_, &r)| r)
        .sum();

    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;

    // Normal approximation
    let mu = n1f * n2f / 2.0;
    let sigma = (n1f * n2f * (n1f + n2f + 1.0) / 12.0).sqrt();

    if sigma < 1e-15 {
        return (u1, 0.0, 1.0);
    }

    // Continuity correction
    let z = (u1 - mu - 0.5_f64.copysign(u1 - mu)) / sigma;

    // Two-sided p-value from standard normal
    use statrs::distribution::Normal;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p = 2.0 * (1.0 - normal.cdf(z.abs()));

    (u1, z, p)
}

/// Benjamini-Hochberg FDR correction (in-place).
///
/// Modifies p-values in place to adjusted p-values.
/// Maintains monotonicity: adjusted[i] >= adjusted[j] if p[i] >= p[j].
pub fn bh_adjust(pvalues: &mut [f64]) {
    let n = pvalues.len();
    if n == 0 {
        return;
    }

    // Sort indices by p-value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Adjust from largest to smallest
    let mut cummin = f64::INFINITY;
    for i in (0..n).rev() {
        let idx = indices[i];
        let rank = (i + 1) as f64;
        let adjusted = (pvalues[idx] * n as f64 / rank).min(1.0);
        cummin = cummin.min(adjusted);
        pvalues[idx] = cummin;
    }
}

/// Log2 counts per million.
///
/// Formula: log2((count + prior) / (lib_size + 2*prior) * 1e6)
/// Matches edgeR::cpm(y, log=TRUE, prior.count=prior).
#[inline]
pub fn log2_cpm(count: f64, lib_size: f64, prior: f64) -> f64 {
    ((count + prior) / (lib_size + 2.0 * prior) * 1e6).log2()
}

/// Log2 fold change between two groups.
///
/// If `is_logged` is true, input is log2-transformed (e.g. log2 CPM):
/// returns `mean_b - mean_a` (difference on log scale IS the log2 fold change).
///
/// If `is_logged` is false, input is raw/linear-scale counts:
/// returns `log2((mean_b + 1) / (mean_a + 1))` with a pseudocount of 1
/// to protect against zero means.
pub fn log2_fold_change(group_a: &[f64], group_b: &[f64], is_logged: bool) -> f64 {
    let mean_a: f64 = group_a.iter().sum::<f64>() / group_a.len() as f64;
    let mean_b: f64 = group_b.iter().sum::<f64>() / group_b.len() as f64;

    if is_logged {
        mean_b - mean_a
    } else {
        (mean_b + 1.0).log2() - (mean_a + 1.0).log2()
    }
}

// ═══════════════════════════════════════════════════════════
// Empirical Bayes moderated t-statistics (Smyth 2004)
// ═══════════════════════════════════════════════════════════

/// Digamma function (logarithmic derivative of the gamma function)
/// using asymptotic expansion with recurrence relation.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Use recurrence psi(x) = psi(x+1) - 1/x to shift x into asymptotic range
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion for large x
    result += x.ln() - 0.5 / x;
    let x2 = 1.0 / (x * x);
    result -= x2
        * (1.0 / 12.0
            - x2 * (1.0 / 120.0
                - x2 * (1.0 / 252.0 - x2 * (1.0 / 240.0 - x2 / 132.0))));
    result
}

/// Trigamma function (derivative of digamma) using asymptotic expansion
/// with recurrence relation.
///
/// Uses the asymptotic series:
/// psi_1(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - ...
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Use recurrence psi_1(x) = psi_1(x+1) + 1/x^2 to shift x into asymptotic range
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic expansion for large x (Bernoulli number series)
    // psi_1(x) = 1/x + 1/(2x^2) + B_2/x^3 + B_4/x^5 + B_6/x^7 + ...
    // B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66
    let ix = 1.0 / x;
    let ix2 = ix * ix;
    result += ix + ix2 * 0.5
        + ix2 * ix * (1.0 / 6.0
            - ix2 * (1.0 / 30.0
                - ix2 * (1.0 / 42.0
                    - ix2 * (1.0 / 30.0
                        - ix2 * 5.0 / 66.0))));
    result
}

/// Inverse trigamma via Newton's method.
/// Finds x such that trigamma(x) = y.
fn inv_trigamma(y: f64) -> f64 {
    let mut x = if y > 1e-4 { 1.0 / y.sqrt() } else { 100.0 };
    for _ in 0..50 {
        let tg = trigamma(x);
        // Compute polygamma(2, x) (tetragamma) for Newton step derivative
        let mut pg2 = 0.0;
        let mut t = x;
        while t < 8.0 {
            pg2 += 2.0 / (t * t * t);
            t += 1.0;
        }
        let t2 = 1.0 / (t * t);
        pg2 -= (2.0 / (t * t * t)) * (1.0 + 1.5 / t + t2 * (1.0 - t2 * (1.0 / 3.0)));
        pg2 = -pg2; // polygamma(2) is negative for positive x

        let delta = (tg - y) / pg2;
        x -= delta;
        x = x.max(1e-10);
        if delta.abs() < 1e-12 {
            break;
        }
    }
    x
}

/// Fit empirical Bayes prior (d0, s0^2) from gene-wise variances.
/// Implements limma's squeezeVar (Smyth 2004).
///
/// Returns `(d0, s0_sq)` where:
/// - `d0` is the prior degrees of freedom
/// - `s0_sq` is the prior variance
///
/// `variances` are gene-wise sample variances and `df` is the residual
/// degrees of freedom per gene (typically n_a + n_b - 2 for two groups).
pub fn fit_ebayes_prior(variances: &[f64], df: f64) -> (f64, f64) {
    let log_vars: Vec<f64> = variances
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| v.ln())
        .collect();

    let n = log_vars.len() as f64;
    if n < 3.0 {
        return (0.0, 1.0); // not enough genes for estimation
    }

    let mean_logvar = log_vars.iter().sum::<f64>() / n;
    let var_logvar = log_vars
        .iter()
        .map(|&z| (z - mean_logvar).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let half_df = df / 2.0;
    let tg_half_df = trigamma(half_df);

    let tg_d0_half = (var_logvar - tg_half_df).max(1e-10);
    let d0_half = inv_trigamma(tg_d0_half);
    let d0 = 2.0 * d0_half;

    let log_s0_sq =
        mean_logvar - digamma(half_df) + half_df.ln() + digamma(d0_half) - d0_half.ln();
    let s0_sq = log_s0_sq.exp();

    (d0, s0_sq)
}

/// Shrink gene-wise variances toward the empirical Bayes prior.
///
/// For each gene, the moderated variance is a weighted average of the
/// gene-wise variance (`s_sq`) and the prior variance (`s0_sq`), weighted
/// by their respective degrees of freedom.
pub fn moderate_variances(variances: &[f64], df: f64, d0: f64, s0_sq: f64) -> Vec<f64> {
    variances
        .iter()
        .map(|&s_sq| (d0 * s0_sq + df * s_sq) / (d0 + df))
        .collect()
}

/// Two-sided p-value from Student's t distribution using statrs.
pub fn students_t_cdf(t: f64, df: f64) -> f64 {
    if df > 0.0 && df.is_finite() && t.is_finite() {
        match StudentsT::new(0.0, 1.0, df) {
            Ok(dist) => dist.cdf(t),
            Err(_) => 0.5, // fallback for invalid df
        }
    } else {
        0.5
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_welford_mean_var() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, var) = welford_mean_var(&data);
        assert_relative_eq!(mean, 5.0, epsilon = 1e-10);
        assert_relative_eq!(var, 4.571428571428571, epsilon = 1e-10);
    }

    #[test]
    fn test_welford_single() {
        let (mean, var) = welford_mean_var(&[42.0]);
        assert_relative_eq!(mean, 42.0, epsilon = 1e-10);
        assert_relative_eq!(var, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_welford_empty() {
        let (mean, var) = welford_mean_var(&[]);
        assert!(mean.is_nan());
        assert!(var.is_nan());
    }

    #[test]
    fn test_welch_t_test_equal_means() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (t, _df, p) = welch_t_test(&a, &b);
        assert_relative_eq!(t, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_welch_t_test_different_means() {
        let a = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (t, _df, p) = welch_t_test(&a, &b);
        assert!(t > 0.0, "t should be positive when group1 > group2");
        assert!(p < 0.001, "p should be very small: {}", p);
    }

    #[test]
    fn test_bh_adjust() {
        let mut pvals = vec![0.01, 0.04, 0.03, 0.20];
        bh_adjust(&mut pvals);
        // All adjusted values should be >= original
        assert!(pvals[0] >= 0.01);
        // Monotonicity: sorted adjusted should be non-decreasing
        let mut sorted_pairs: Vec<(f64, f64)> = vec![0.01, 0.04, 0.03, 0.20]
            .into_iter()
            .zip(pvals.iter().copied())
            .collect();
        sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 1..sorted_pairs.len() {
            assert!(sorted_pairs[i].1 >= sorted_pairs[i - 1].1);
        }
    }

    #[test]
    fn test_log2_cpm() {
        let val = log2_cpm(100.0, 1_000_000.0, 1.0);
        // log2((101) / (1000002) * 1e6) ≈ log2(100.9998) ≈ 6.658
        assert!(val > 6.0 && val < 7.0, "log2_cpm = {}", val);
    }

    #[test]
    fn test_wilcoxon_equal() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_u, _z, p) = wilcoxon_rank_sum(&a, &b);
        assert!(p > 0.5, "p should be large for identical groups: {}", p);
    }

    #[test]
    fn test_wilcoxon_different() {
        let a = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_u, _z, p) = wilcoxon_rank_sum(&a, &b);
        assert!(p < 0.05, "p should be small: {}", p);
    }

    #[test]
    fn test_log2_fold_change_raw() {
        let a = vec![100.0, 100.0, 100.0];
        let b = vec![10.0, 10.0, 10.0];
        let lfc = log2_fold_change(&a, &b, false);
        // log2(11) - log2(101) ≈ 3.459 - 6.658 ≈ -3.199
        assert!(lfc < -3.0 && lfc > -3.5, "lfc = {}", lfc);
    }

    #[test]
    fn test_log2_fold_change_logged() {
        // Simulated log2 CPM values
        let a = vec![5.0, 5.5, 5.2];
        let b = vec![8.0, 8.3, 8.1];
        let lfc = log2_fold_change(&a, &b, true);
        // mean_b - mean_a = 8.133 - 5.233 ≈ 2.9
        assert!(lfc > 2.5 && lfc < 3.5, "lfc = {}", lfc);
    }

    // ── Empirical Bayes tests ──

    #[test]
    fn test_digamma_known_values() {
        // psi(1) = -gamma (Euler-Mascheroni constant)
        assert!((digamma(1.0) - (-0.5772156649)).abs() < 1e-6);
        // psi(0.5) = -gamma - 2*ln(2)
        assert!((digamma(0.5) - (-1.9635100260)).abs() < 1e-4);
    }

    #[test]
    fn test_trigamma_known_values() {
        // psi_1(1) = pi^2/6
        assert!(
            (trigamma(1.0) - 1.6449340668).abs() < 1e-6,
            "trigamma(1.0) = {}, expected ~1.6449340668",
            trigamma(1.0)
        );
    }

    #[test]
    fn test_moderated_variance_shrinkage() {
        let variances = vec![0.01, 100.0, 1.0, 1.0, 1.0, 0.5, 2.0, 0.8, 1.2, 0.9];
        let (d0, s0_sq) = fit_ebayes_prior(&variances, 4.0);
        assert!(d0 > 0.0, "Prior df should be positive, got {}", d0);
        assert!(s0_sq > 0.0, "Prior variance should be positive, got {}", s0_sq);

        let moderated = moderate_variances(&variances, 4.0, d0, s0_sq);
        assert!(
            moderated[0] > variances[0],
            "Small variance ({}) should increase toward prior (got {})",
            variances[0],
            moderated[0]
        );
        assert!(
            moderated[1] < variances[1],
            "Large variance ({}) should decrease toward prior (got {})",
            variances[1],
            moderated[1]
        );
    }

    #[test]
    fn test_students_t_cdf_symmetry() {
        // CDF at 0 for any df should be 0.5
        assert_relative_eq!(students_t_cdf(0.0, 5.0), 0.5, epsilon = 1e-10);
        // CDF for large positive t should be close to 1
        assert!(students_t_cdf(10.0, 5.0) > 0.99);
        // CDF for large negative t should be close to 0
        assert!(students_t_cdf(-10.0, 5.0) < 0.01);
    }
}
