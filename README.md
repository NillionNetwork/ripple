<h1 align="center">Ripple
  <a href="https://github.com/NillionNetwork/ripple/actions/workflows/ci-build.yml"><img src="https://github.com/NillionNetwork/ripple/workflows/ci-build/badge.svg"></a>
  <a href="https://github.com/NillionNetwork/ripple/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>

</h1>
<p align="center">
  <img src="./assets/ripple.png" height="20%" width="20%">
</p>

# Building

For `x86_64`-based machines running Unix-like OSes:
```bash
❯❯ cargo b --release --features x86
```

For Apple Silicon or `aarch64`-based machines running Unix-like OSes:
```bash
❯❯ cargo b --release --features aarch64
```

# Running

```bash
❯❯ cargo run --release --bin correlation
❯❯ cargo run --release --bin euclidean
❯❯ cargo run --release --bin lr
```

## Disclaimer
This is software for a research prototype and not production-ready code.
