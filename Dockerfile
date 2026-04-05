# ── Build stage ──────────────────────────────────────────────────────────────
# Uses the official Rust image so the toolchain is pre-installed.
# bookworm = Debian 12 (latest stable), matches the runtime base.
FROM rust:1.75-bookworm AS builder

WORKDIR /build

# Copy manifest files first so cargo can cache dependency compilation.
# Re-running `cargo build` after a source-only change will skip this layer.
COPY Cargo.toml Cargo.lock ./

# Copy source and bench directories needed for a full release build.
COPY src/ src/
COPY benches/ benches/

# Build in release mode.
# strip is also configured in Cargo.toml [profile.release] but calling it
# here explicitly keeps the intent clear for anyone reading the Dockerfile.
RUN cargo build --release && strip target/release/rustpipe

# ── Runtime stage ─────────────────────────────────────────────────────────────
# debian:bookworm-slim gives us a minimal libc environment (~30 MB).
# No Rust toolchain, no build deps — only the compiled binary lands here.
FROM debian:bookworm-slim

LABEL org.opencontainers.image.source="https://github.com/jwg054000/RustPipe"
LABEL org.opencontainers.image.description="Fast downstream RNA-seq analysis in Rust"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.version="0.1.0"

# Copy only the stripped binary from the build stage.
COPY --from=builder /build/target/release/rustpipe /usr/local/bin/rustpipe

# Smoke-test: fails the build if the binary is missing or won't execute.
RUN rustpipe --version

ENTRYPOINT ["rustpipe"]
