//! RustPipe — Fast downstream RNA-seq analysis in Rust
//!
//! The companion to RustQC. Handles everything after quality control:
//! normalization, differential expression, PCA, and pathway enrichment.
//!
//! Built for speed with automatic multi-core parallelism.

use clap::{Parser, Subcommand};
use log::info;
use std::path::PathBuf;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod convert;
mod de;
mod enrich;
mod filter;
mod hvg;
mod io;
mod normalize;
mod pca;
mod pipeline;
mod stats;

#[derive(Parser)]
#[command(
    name = "rustpipe",
    version,
    about = "Fast downstream RNA-seq analysis in Rust",
    long_about = "RustPipe picks up where RustQC leaves off.\n\n\
        Normalize → DE → PCA → GSEA in seconds on Apache Parquet.\n\n\
        https://github.com/jwg054000/RustPipe"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose logging (repeat for more: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Number of threads (default: all available)
    #[arg(long, global = true)]
    threads: Option<usize>,

    /// Random seed for reproducibility (default: 42)
    #[arg(long, global = true, default_value = "42")]
    seed: u64,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert CSV expression matrix to Parquet
    Convert {
        /// Input CSV file (genes × samples, first column = gene names)
        #[arg(short, long)]
        input: PathBuf,

        /// Output Parquet file
        #[arg(short, long)]
        output: PathBuf,

        /// Compression level (1-22, default 3)
        #[arg(long, default_value = "3")]
        compression_level: u32,

        /// Column separator character (auto-detected if omitted)
        #[arg(long)]
        separator: Option<char>,
    },

    /// TMM normalization + log2 CPM transform
    Normalize {
        /// Input Parquet (genes × samples)
        #[arg(short, long)]
        input: PathBuf,

        /// Output normalized Parquet
        #[arg(short, long)]
        output: PathBuf,

        /// Prior count for log2 CPM (default 1.0)
        #[arg(long, default_value = "1.0")]
        prior_count: f64,
    },

    /// Differential expression (Welch t-test, Wilcoxon rank-sum, or moderated t)
    De {
        /// Input normalized Parquet
        #[arg(short, long)]
        input: PathBuf,

        /// Metadata Parquet with sample grouping
        #[arg(short, long)]
        metadata: PathBuf,

        /// Column in metadata for group comparison
        #[arg(short, long)]
        group_col: String,

        /// Output directory (one Parquet per contrast)
        #[arg(short, long)]
        output: PathBuf,

        /// Test method: "welch" (default, bulk), "wilcoxon" (single-cell), or "moderated" (empirical Bayes)
        #[arg(long, default_value = "welch")]
        method: String,
    },

    /// Principal component analysis (randomized SVD)
    Pca {
        /// Input normalized Parquet
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Number of principal components
        #[arg(long, default_value = "50")]
        n_pcs: usize,

        /// Power iteration rounds for rSVD accuracy (default 3)
        #[arg(long, default_value = "3")]
        n_power_iter: usize,
    },

    /// Gene set enrichment analysis (permutation-based)
    Enrich {
        /// Directory containing DE results (Parquet files)
        #[arg(short, long)]
        de_results: PathBuf,

        /// Gene sets file (JSON or GMT format, auto-detected by extension)
        #[arg(short, long)]
        gene_sets: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Number of permutations for p-value estimation (default 1000)
        #[arg(long, default_value = "1000")]
        n_permutations: usize,
    },

    /// Select highly variable genes (VST method)
    Hvg {
        /// Input expression Parquet (genes × samples)
        #[arg(short, long)]
        input: PathBuf,

        /// Output filtered Parquet (HVGs × samples)
        #[arg(short, long)]
        output: PathBuf,

        /// Number of top HVGs to select (default 2000)
        #[arg(long, default_value = "2000")]
        n_top_genes: usize,
    },

    /// Filter low-count genes (edgeR filterByExpr defaults)
    Filter {
        /// Input expression Parquet (genes × samples)
        #[arg(short, long)]
        input: PathBuf,

        /// Output filtered Parquet
        #[arg(short, long)]
        output: PathBuf,

        /// Minimum count threshold per gene per sample (default 10)
        #[arg(long, default_value = "10")]
        min_count: f64,

        /// Minimum number of samples that must meet the count threshold (default 3)
        #[arg(long, default_value = "3")]
        min_samples: usize,
    },

    /// Run full pipeline: normalize → DE → PCA → enrich
    Pipeline {
        /// Input expression Parquet (genes × samples)
        #[arg(short, long)]
        input: PathBuf,

        /// Metadata Parquet
        #[arg(short, long)]
        metadata: PathBuf,

        /// Group column for DE
        #[arg(short, long)]
        group_col: String,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Gene sets file (optional, skips enrichment if absent)
        #[arg(long)]
        gene_sets: Option<PathBuf>,

        /// Number of PCs
        #[arg(long, default_value = "50")]
        n_pcs: usize,

        /// Skip normalization (input already normalized)
        #[arg(long)]
        skip_normalize: bool,

        /// Number of HVGs to select after normalization (optional, skips HVG if absent)
        #[arg(long)]
        n_hvg: Option<usize>,

        /// Minimum count for low-count gene filtering (optional, skips filtering if absent)
        #[arg(long)]
        min_count: Option<f64>,

        /// Minimum samples for low-count gene filtering (default 3, used with --min-count)
        #[arg(long)]
        min_samples: Option<usize>,
    },

    /// Run benchmarks on synthetic data
    Bench {
        /// Number of genes
        #[arg(long, default_value = "18168")]
        n_genes: usize,

        /// Number of samples
        #[arg(long, default_value = "800")]
        n_samples: usize,

        /// Output directory for benchmark results
        #[arg(short, long, default_value = "bench_output")]
        output: PathBuf,
    },
}

fn init_logger(verbosity: u8) {
    let level = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
        .format_timestamp_millis()
        .init();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_logger(cli.verbose);

    // Configure thread pool if requested
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    let seed = cli.seed;

    match cli.command {
        Commands::Convert {
            input,
            output,
            compression_level,
            separator,
        } => {
            info!("Converting {} → {}", input.display(), output.display());
            let sep = separator.map(|c| c as u8);
            convert::csv_to_parquet(&input, &output, compression_level, sep)?;
        }

        Commands::Normalize {
            input,
            output,
            prior_count,
        } => {
            info!("Normalizing {}", input.display());
            normalize::tmm_normalize(&input, &output, prior_count)?;
        }

        Commands::De {
            input,
            metadata,
            group_col,
            output,
            method,
        } => {
            info!("Differential expression: {}", method);
            de::run_de(&input, &metadata, &group_col, &output, &method)?;
        }

        Commands::Pca {
            input,
            output,
            n_pcs,
            n_power_iter,
        } => {
            info!(
                "PCA: {} components, {} power iterations",
                n_pcs, n_power_iter
            );
            let _pca_result = pca::run_pca(&input, &output, n_pcs, n_power_iter)?;
        }

        Commands::Enrich {
            de_results,
            gene_sets,
            output,
            n_permutations,
        } => {
            info!("Enrichment analysis ({} permutations)", n_permutations);
            enrich::run_enrichment(&de_results, &gene_sets, &output, n_permutations, seed)?;
        }

        Commands::Hvg {
            input,
            output,
            n_top_genes,
        } => {
            info!("HVG selection: top {} genes", n_top_genes);
            hvg::run_hvg(&input, &output, n_top_genes)?;
        }

        Commands::Filter {
            input,
            output,
            min_count,
            min_samples,
        } => {
            info!(
                "Filtering low-count genes (min_count={}, min_samples={})",
                min_count, min_samples
            );
            filter::run_filter(&input, &output, min_count, min_samples)?;
        }

        Commands::Pipeline {
            input,
            metadata,
            group_col,
            output,
            gene_sets,
            n_pcs,
            skip_normalize,
            n_hvg,
            min_count,
            min_samples,
        } => {
            info!("Running full pipeline");
            pipeline::run_full(
                &input,
                &metadata,
                &group_col,
                &output,
                gene_sets.as_deref(),
                n_pcs,
                skip_normalize,
                n_hvg,
                min_count,
                min_samples,
                seed,
            )?;
        }

        Commands::Bench {
            n_genes,
            n_samples,
            output,
        } => {
            info!("Benchmark: {} genes × {} samples", n_genes, n_samples);
            pipeline::bench(n_genes, n_samples, &output)?;
        }
    }

    Ok(())
}
