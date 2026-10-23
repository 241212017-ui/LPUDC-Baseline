# LPUDC Reproduction Report

- **Dataset:** synthetic_tiny
- **Split:** test
- **Images evaluated:** 2
- **Sec/image (ours):** 0.13032281398773193

## Direction of metrics

| Metric | Better when |
|--------|-------------|
| PSNR | higher |
| SSIM | higher |
| LPIPS | lower |
| DISTS | lower |

## Comparison

| Metric | Paper (reference) | Our result | Absolute difference |
|--------|-------------------:|-----------:|--------------------:|
| (all) | No paper entry for this dataset name | see metrics.json | n/a |

## Classification of values

- **PAPER FACT:** the "Paper (reference)" column, from
  Ding et al., Neurocomputing 655 (2025) 131442, Tables 1-2.
- **OUR EXPERIMENTAL RESULT:** the "Our result" column, measured
  by `test.py` on our checkpoint.
- **IMPLEMENTATION ASSUMPTION:** architecture widths and unspecified
  training details are documented in `docs/PAPER_SPECIFICATION.md`.

## Possible reasons for difference

1. **Missing implementation details.** The paper does not specify channel
   widths, number of DRBs, encoder/decoder depths, inverse pixel-shuffle
   factor, or the exact DGF network layout. Our choices are documented
   assumptions and will not match the authors' private code exactly.
2. **Laplacian pyramid kernels.** The paper says "fixed-kernel weighted
   average" but does not give the kernel. We use the classical 5x5
   binomial kernel.
3. **Loss reduction.** The paper writes L1 as a sum over pixels; we use
   mean L1 for scale stability.
4. **Training schedule details.** Cosine T_max definition, weight decay,
   initialization, and validation protocol are unspecified.
5. **Dataset files / split layout.** POLED/TOLED release folder layouts
   vary; our loader matches by file stem.
6. **Hardware / precision.** Paper: single RTX 3090, no mention of AMP.
   Our run hardware and precision settings may differ.
7. **Randomness.** The paper does not report a seed.
8. **Metric implementations.** LPIPS backbone and DISTS version are not
   stated; slight metric-implementation differences are expected.
9. **Training length.** Reproducing 300K full-resolution iterations
   requires substantial GPU time; shorter runs will not match.

## Integrity statement

This report does **not** claim exact reproduction. Paper numbers are
reference targets only and are never hardcoded into evaluation code.
If our measured values differ, the difference is reported honestly above.
