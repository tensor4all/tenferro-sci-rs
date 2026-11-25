<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to tenferro-sci-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `tenferro-sci-rs`. The repository's own rules are in
[REPOSITORY_RULES.md](REPOSITORY_RULES.md).

## Local facts

- **MSRV**: the same `rust-version` as tenferro-rs (1.96 at the time of writing), set once in the root `Cargo.toml`.
- **Feature combinations CI builds**: default features. GPU features are added with the first GPU path. No CI lane
  builds `specials` with `--no-default-features` yet (#30); run it locally (below).
- **Platforms claimed**: Linux x86_64, tested in CI. Other platforms are not claimed yet.
- **Local gate**: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `cargo clippy -p specials --no-default-features --all-targets -- -D warnings`,
  `cargo test -p specials --no-default-features`, and `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`.

## Anything here that departs from the shared rules

The repository has no crate yet, so the `fmt`, `clippy`, `test`, `msrv` and `docs` jobs skip until the first PR adds
the root `Cargo.toml`. That PR makes them run.
