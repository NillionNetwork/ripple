<h1 align="center">Ripple: Accelerating Programmable Bootstraps for FHE with Wavelet Approximations
  <a href="https://github.com/NillionNetwork/ripple/actions/workflows/ci-build.yml"><img src="https://github.com/NillionNetwork/ripple/workflows/ci-build/badge.svg"></a>
  <a href="https://github.com/NillionNetwork/ripple/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache-blue.svg"></a>

</h1>
<p align="center">
  <img src="./assets/ripple.png" height="20%" width="20%">
</p>

## How to cite this work
The preprint can be accessed [here](https://eprint.iacr.org/2024/XXX); you can
cite this work as follows:
```bibtex
@Misc{EPRINT:GUMVT24,
  author =       "Charles Gouert and
                  Mehmet Ugurbil and
                  Dimitris Mouris and
                  Miguel de Vega and
                  Nektarios Georgios Tsoutsos",
  title =        "{Ripple: Accelerating Programmable Bootstraps for FHE with Wavelet Approximations}",
  year =         2024,
  howpublished = "Cryptology ePrint Archive, Report 2024/XXX",
  note =         "\url{https://eprint.iacr.org/2024/XXX}",
}
```

## Building & Running

This repository comprises multiple binaries located at the [src](https://github.com/NillionNetwork/ripple/tree/main/src) directory. For each application, we have different variants, namely: plaintext, Haar DWT, Biorthogonal DWT, and quantization.


### Building
For `x86_64`-based machines running Unix-like OSes:
```bash
❯❯ cargo b --release --features x86
```

For Apple Silicon or `aarch64`-based machines running Unix-like OSes:
```bash
❯❯ cargo b --release --features aarch64
```

### Running

Example:
```bash
❯❯ cargo run --release --bin lr_haar
```

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [TFHE-rs](https://github.com/zama-ai/tfhe-rs) and [DWT](https://github.com/stainless-steel/dwt).
