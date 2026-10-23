# LPUDC — Laplacian Pyramid Neural Network for Under-Display Camera Image Restoration

Research-grade reimplementation of:

> **LPUDC: A Laplacian pyramid neural network for restoring images from under-display cameras**
> Zhenyan Ding, Zhixiang Fang, Zhengning Wang, Lehan Ding, Zhenni Zeng, Xinglong Luo, Shaoqin Yuan, Haoran Ma, Binquan Leng
> *Neurocomputing* 655 (2025) 131442. DOI: [10.1016/j.neucom.2025.131442](https://doi.org/10.1016/j.neucom.2025.131442)

This codebase reconstructs the method **faithfully from the paper**, implements the full training / validation / testing pipeline, and stays **dataset-independent** so you can swap POLED/TOLED for your own paired restoration data.

---

## Research integrity rules (enforced in code)

1. Paper-reported metrics are **reference targets only** — never hardcoded into scoring.
2. Every architecture choice the paper does **not** specify is documented as an `IMPLEMENTATION ASSUMPTION` in `docs/PAPER_SPECIFICATION.md`.
3. Evaluation code never fabricates LPIPS/DISTS if the package/backbone is unavailable (reports `null`).
4. Train / val / test splits are strict; no test images during training.
5. Configs + seeds are saved with every run.

---

## Architecture

```
INPUT DEGRADED IMAGE
        |
        v
LAPLACIAN PYRAMID DECOMPOSITION   (default 3 levels)
        |
        +------------------------------+
        |                              |
        v                              v
LOW-FREQUENCY BRANCH             HIGH-FREQUENCY BRANCH
  (coarsest LP level)              (each HF residual level)
        |                              ^
        |                              |
        v                              |
LOW-FREQUENCY RESTORATION         DGF GUIDANCE
  3x3 conv + LeakyReLU              ^
  + DRB                             |
  + encoder-decoder                 |
  + symmetric reconstruction        |
        |                              |
        +----------> DGF --------------+
                       |
                       v
                HIGH-FREQUENCY
                  ENHANCEMENT
            inverse pixel shuffle
            3x3 conv + DRBs + HRNet trunk
            residual fusion with LP HF
                       |
                       v
             PYRAMID RECONSTRUCTION
                       |
                       v
                 RESTORED IMAGE

LOSS:  L = L1(Y, X) + 0.5 * sum_i L1(Î_i, I_i)
```

### What is PAPER FACT vs ASSUMPTION

| Component | Paper states | Our implementation |
|-----------|--------------|--------------------|
| Pyramid levels | 3 (primary) | configurable, default 3 |
| LF branch | 3x3 conv + LeakyReLU + DRB + enc-dec + symmetric recon | as specified; widths are assumptions |
| DGF | learnable weighted means + conv replacing linear map; G=LF input, I=LF output; coeffs upsampled to upper level | as specified; internal layout follows Wu et al. CVPR 2018 lineage (paper ref [9]) |
| HF branch | inverse pixel shuffle + 3x3 conv + DRBs + HRNet (serial DRBs + residual) + 3x3 proj fused with LP HF | as specified; "HRNet" = serial DRB trunk (NOT full multi-branch HRNet) |
| Loss | L1 + 0.5 * pyramid L1 | λ=0.5 exactly; mean-reduction (documented) |
| Optim | Adam β1=0.9 β2=0.999 | exact |
| LR | 2e-4 → 1e-6 cosine | exact, stepped per-iteration |
| Batch / iters | 1 / 300K | exact in reproduction configs |
| Augment | random H + V flip | exact |
| Resolution | full 1024x2048 | exact in reproduction configs |
| Hardware | RTX 3090, PyTorch | we print Device/GPU/VRAM; this machine may differ |

Full detail: [`docs/PAPER_SPECIFICATION.md`](docs/PAPER_SPECIFICATION.md)

### Measured model size (this implementation)

| Config | Params | Paper |
|--------|-------:|------:|
| `configs/lpudc_poled.yaml` (3-level) | **4.28M** | ~4.0M |
| `configs/ablations/lpudc_tiny.yaml` | 1.02M | (debug only) |

Inference latency and peak GPU memory are measured by `tools/model_profile.py` on **your** hardware. The paper's ~0.07s on RTX 3090 is a reference, not a claim we reproduce.

---

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
```

Optional:
```bash
pip install thop fvcore   # FLOPs profiling
```

Windows note: if you see an OpenMP conflict, set `KMP_DUPLICATE_LIB_OK=TRUE`.

---

## Dataset preparation

### Paper datasets (POLED / TOLED)

Expected layout (paths come from YAML — never hardcoded):

```
data/POLED/
    train/
        degraded/   # UDC-degraded RGB images
        gt/         # clean ground-truth RGB images
    val/
        degraded/
        gt/
    test/
        degraded/
        gt/
```

Filenames are matched by **stem** (basename without extension).  
POLED and TOLED originate from Zhou et al., CVPR 2021. Place your copies under the roots set in:

- `configs/lpudc_poled.yaml` → `dataset.root: data/POLED`
- `configs/lpudc_toled.yaml` → `dataset.root: data/TOLED`

### Analyze any dataset first

```bash
python tools/dataset_analysis.py --config configs/custom_dataset.yaml
# writes docs/DATASET_REPORT.md + results/dataset_analysis/viz_grid.png
```

If the data is **not** paired degraded→GT image data, the analyzer **stops** and explains why. LPUDC requires a supervised paired restoration setup.

---

## Training

### Reproduction configuration (paper-stated hyperparameters)

```bash
python train.py --config configs/lpudc_poled.yaml
python train.py --config configs/lpudc_toled.yaml
```

Do **not** change paper-stated values in these files. If you hit CUDA OOM:

```bash
python train.py --config configs/lpudc_poled_debug.yaml   # clearly labeled DEBUG
```

The debug config uses crops / fewer iterations / optional AMP and is **not** comparable to paper numbers.

### Resume

```bash
python train.py --config configs/lpudc_poled.yaml --resume checkpoints/lpudc_poled/latest.pth
```

### Outputs

```
checkpoints/<run>/
    config.yaml
    latest.pth
    best_psnr.pth
    best_ssim.pth
logs/<run>/
    config.yaml
    train_log.jsonl
    val_log.jsonl
    metrics.csv
results/<run>/
    visualizations/
        train_iter_*.png          # [DEGRADED][RESTORED][GT]
        pyramid_iter_*/           # per-level GT vs predicted
        guidance_iter_*/          # DGF intermediates
```

Checkpoint payload includes: model, optimizer, scheduler, iteration, best metrics, full config.

---

## Validation / Testing / Evaluation

```bash
# Test split metrics + visualizations
python test.py --config configs/lpudc_poled.yaml \
    --checkpoint checkpoints/lpudc_poled/best_psnr.pth \
    --split test --save-visualizations --save-pyramid

# Evaluate + write docs/REPRODUCTION_REPORT.md (paper vs ours)
python evaluate.py --config configs/lpudc_poled.yaml \
    --checkpoint checkpoints/lpudc_poled/best_psnr.pth --split test
```

### Metrics and direction

| Metric | Direction | Source |
|--------|-----------|--------|
| PSNR | **higher** is better | implemented in `metrics/psnr.py` |
| SSIM | **higher** is better | implemented in `metrics/ssim.py` |
| LPIPS | **lower** is better | `lpips` package (AlexNet default; paper does not specify) |
| DISTS | **lower** is better | official `dists-pytorch` implementation; reports `null` if unavailable |

Outputs: `results/metrics.json`, `results/metrics.csv`, `results/visualizations/`.

### Paper reference targets (NOT used by code)

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|---------|------:|------:|-------:|-------:|
| POLED | 33.99 | 0.9629 | 0.1678 | 0.1399 |
| TOLED | 38.62 | 0.9805 | 0.0893 | 0.0858 |

If our numbers differ, `docs/REPRODUCTION_REPORT.md` records paper value, our value, absolute difference, and possible reasons. We do **not** claim exact reproduction.

---

## Visualization / debugging

Verify the model is learning frequency decomposition, not just a plain CNN:

```bash
# Laplacian pyramid of any image
python tools/visualize_pyramid.py --image path/to/img.png --levels 3

# Model pyramid levels for a dataset sample
python tools/visualize_pyramid.py --config configs/lpudc_poled.yaml \
    --checkpoint checkpoints/lpudc_poled/latest.pth --split val --index 0
```

During training, `pyramid_iter_*` and `guidance_iter_*` folders are written automatically.

---

## Custom dataset usage

1. Arrange your data:

```
your_dataset/
    train/{degraded,gt}/
    val/{degraded,gt}/
    test/{degraded,gt}/
```

2. Edit only the path (and optionally training length):

```yaml
# configs/custom_dataset.yaml
dataset:
  root: /path/to/your_dataset
training:
  iterations: 100000
  batch_size: 1
```

3. Analyze + train + evaluate:

```bash
python scripts/train_custom.py --config configs/custom_dataset.yaml
```

The LPUDC architecture is **unchanged**. Only after this baseline works should you design new modules (e.g. an attention-based fusion to replace DGF).

---

## Ablation studies

Configs in `configs/ablations/`:

| Config | Setting |
|--------|---------|
| `ablation_a_no_multiscale` | No cross-level interaction |
| `ablation_b_concat_no_dgf` | Multi-scale without GF (direct projection/concat) |
| `ablation_c_full_dgf` | Multi-scale + guided filtering (full LPUDC) |
| `ablation_d_levels_{2,3,4}` | Pyramid depth |
| `ablation_e_l1_only` | Content loss only |
| `ablation_e_l1_plus_pyr` | L1 + 0.5 * pyramid loss |

```bash
# Run a subset (edit dataset.root first)
python scripts/run_ablation.py --configs configs/ablations/ablation_c_full_dgf.yaml

# Rebuild table from existing results
python scripts/run_ablation.py --table-only
```

Table: `results/ablations/ablation_table.md` / `.csv`  
Missing metrics are `—` (never fabricated).

## Experimental extension: confidence-gated DGF

`configs/lpudc_poled_3050_confidence_gate.yaml` enables a learned spatial/channel gate
between DGF guidance and high-frequency features. It is an experimental extension,
not part of the paper architecture. It matches the 256x256 AMP debug protocol and
is not comparable to the paper's full-resolution results. The original debug
baseline remains available in `configs/lpudc_poled_3050_debug.yaml`.

```bash
python train.py --config configs/lpudc_poled_3050_confidence_gate.yaml --device cuda
python test.py --config configs/lpudc_poled_3050_confidence_gate.yaml \
    --checkpoint checkpoints/lpudc_poled_3050_confidence_gate/best_psnr.pth \
    --split test --device cuda --no-visualizations \
    --output-dir results/lpudc_poled_3050_confidence_gate/test_eval
```

---

## Model complexity

```bash
python tools/model_profile.py --config configs/lpudc_poled.yaml \
    --image-size 1024 2048 --json-out results/model_profile.json
```

Reports: total/trainable params, model size, FLOPs (if `thop`/`fvcore` installed), measured latency, peak GPU memory.

---

## Tests

```bash
pytest -q
```

Covers: pyramid dims + reconstruction, DRB, LF branch, DGF shapes/gradients, HF branch, full LPUDC forward (fusion modes 0/1/2, levels 2/3/4, odd sizes), end-to-end gradients, loss components, metrics, dataset pairing, checkpoint save/load, and a **tiny overfit sanity check**.

---

## GPU / memory safety

Every run prints:

```
Device : ...
GPU    : ...
VRAM   : ...
```

If full-resolution training causes CUDA OOM:

1. The OOM is reported (not silently swallowed).
2. Explanation: paper uses 1024x2048 with batch 1 on RTX 3090.
3. Use `configs/lpudc_poled_debug.yaml` (crops, fewer iters, optional AMP, grad checkpointing).
4. The official reproduction config is **left unchanged**.

Engineering options (AMP, gradient accumulation, gradient checkpointing) are clearly labeled as **not in the paper**.

---

## Project structure

```
LPUDC/
├── configs/
│   ├── lpudc_poled.yaml          # paper reproduction (POLED)
│   ├── lpudc_toled.yaml          # paper reproduction (TOLED)
│   ├── lpudc_poled_debug.yaml    # low-memory DEBUG (not paper setup)
│   ├── custom_dataset.yaml
│   └── ablations/
├── data/
│   ├── dataset.py
│   ├── transforms.py
│   └── dataloader.py
├── models/
│   ├── lpudc.py                  # integrated network
│   ├── laplacian_pyramid.py
│   ├── blocks.py                 # DRB, conv blocks
│   ├── low_frequency.py
│   ├── dgf.py                    # Deep Guided Filtering
│   ├── high_frequency.py
│   └── reconstruction.py
├── losses/
│   └── lpudc_loss.py
├── metrics/
│   ├── psnr.py
│   ├── ssim.py
│   ├── lpips_metric.py
│   └── dists_metric.py
├── tools/
│   ├── model_profile.py
│   ├── dataset_analysis.py
│   ├── visualize_pyramid.py
│   ├── visualize_utils.py
│   └── make_synthetic_dataset.py
├── scripts/
│   ├── train_custom.py
│   └── run_ablation.py
├── tests/
│   └── test_lpudc.py
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
├── pytest.ini
├── docs/
│   ├── PAPER_SPECIFICATION.md
│   ├── DATASET_REPORT.md         # after you run dataset_analysis
│   └── REPRODUCTION_REPORT.md    # after evaluate.py
├── checkpoints/
├── logs/
├── results/
└── visualizations/
```

---

## Known differences from the paper

1. **Unspecified architecture widths** — encoder/decoder channels, DRB counts, HRNet trunk depth, inverse pixel-shuffle factor, DGF internal layout are engineering assumptions (see `docs/PAPER_SPECIFICATION.md`).
2. **"HRNet" interpretation** — implemented as a serial-DRB high-resolution trunk with global residual, as described in the paper text; **not** the full multi-branch HRNet of Wang et al.
3. **Laplacian kernels** — classical 5×5 binomial (paper: unspecified "fixed-kernel weighted average").
4. **Loss reduction** — mean L1 instead of raw pixel sum (scale stability); λ=0.5 exactly as stated.
5. **Metric backbones** — LPIPS default `alex` (paper unspecified); DISTS is a self-contained reimplementation.
6. **Params** — our 3-level model measures **4.29M** vs paper **4.0M**.
7. **No claim of exact reproduction** until a full 300K-iteration run on POLED/TOLED with matching hardware produces comparable numbers.

---

## Future research directions (not in the baseline)

Per the paper's own outlook and standard restoration research, clean extension points:

- Different DGF variants / attention-based fusion modules
- Transformer / Mamba blocks in the HF trunk
- Frequency-domain processing
- SSIM / Charbonnier / edge / perceptual losses
- Diffusion-based refinement

**Do not mix these into the baseline.** Reproduce LPUDC first; then branch.

---

## Citation

```bibtex
@article{ding2025lpudc,
  title={LPUDC: A Laplacian pyramid neural network for restoring images from under-display cameras},
  author={Ding, Zhenyan and Fang, Zhixiang and Wang, Zhengning and Ding, Lehan and Zeng, Zhenni and Luo, Xinglong and Yuan, Shaoqin and Ma, Haoran and Leng, Binquan},
  journal={Neurocomputing},
  volume={655},
  pages={131442},
  year={2025},
  publisher={Elsevier}
}
```
