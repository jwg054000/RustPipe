//! Criterion benchmarks for core pipeline operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_distr::StandardNormal;

// Re-implement stats functions here since we can't import from the binary crate.
// In a library crate refactor, these would come from `rustpipe::stats`.

fn welford_mean_var(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 2 {
        return (data.first().copied().unwrap_or(f64::NAN), 0.0);
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

fn bh_adjust(pvalues: &mut [f64]) {
    let n = pvalues.len();
    if n == 0 {
        return;
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cummin = f64::INFINITY;
    for i in (0..n).rev() {
        let idx = indices[i];
        let rank = (i + 1) as f64;
        let adjusted = (pvalues[idx] * n as f64 / rank).min(1.0);
        cummin = cummin.min(adjusted);
        pvalues[idx] = cummin;
    }
}

fn bench_welford(c: &mut Criterion) {
    let mut group = c.benchmark_group("welford_mean_var");

    for size in [100, 1_000, 10_000, 100_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| welford_mean_var(black_box(data)));
        });
    }

    group.finish();
}

fn bench_bh_adjust(c: &mut Criterion) {
    let mut group = c.benchmark_group("bh_adjust");

    for size in [100, 1_000, 18_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let pvals: Vec<f64> = (0..size).map(|_| rng.gen::<f64>()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &pvals, |b, pvals| {
            b.iter(|| {
                let mut p = pvals.clone();
                bh_adjust(black_box(&mut p));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_welford, bench_bh_adjust);
criterion_main!(benches);
