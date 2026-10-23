# LPUDC Paper Specification

**Paper:** LPUDC: A Laplacian pyramid neural network for restoring images from under-display cameras
**Authors:** Zhenyan Ding, Zhixiang Fang, Zhengning Wang, Lehan Ding, Zhenni Zeng, Xinglong Luo, Shaoqin Yuan, Haoran Ma, Binquan Leng
**Venue:** Neurocomputing 655 (2025) 131442
**DOI:** https://doi.org/10.1016/j.neucom.2025.131442
**Source used:** `C:\Users\Razzak\OneDrive\Desktop\RESEARCH PAPER\main.pdf` (text extracted to `paper_extracted.txt`)

---

## A. Details Explicitly Stated in the Paper

### Problem definition
- Images captured by under-display cameras (UDCs) suffer from severe degradation because light passes through multiple display material layers (OLED emitting layers, polarizers, encapsulation layers).
- Two degradation families:
  - **Low-frequency:** light attenuation / brightness reduction, color shift / color distortion (caused mainly by transparent layers).
  - **High-frequency:** diffraction artifacts, structural noise, edge blur / ghosting / striping (caused by the OLED layer's light-emitting units and opaque wiring).
- Classical UDC physical model (Zhou et al.): `y = (γx) * k + n`, where `γ` is attenuation, `k` is PSF, `n` is noise. LPUDC is a **PSF-free** method.

### Input / output format
- Input: degraded UDC image.
- Output: restored image.
- Paired supervised training against clean ground-truth images.
- Datasets used in the paper are RGB images.

### LPUDC architecture (high level)
- Laplacian-pyramid-based multi-scale network.
- Decomposes the input image into high- and low-frequency components and processes them with specialized branches.
- Components:
  1. **Low-Frequency Branch** — encoder–decoder for global degradations (brightness, color shift).
  2. **Deep Guided Filtering (DGF) Module** — cross-scale feature fusion; low-resolution output guides high-level feature extraction.
  3. **High-Frequency Branch** — restores edge blur and diffraction artifacts.
- Each pyramid level is connected by a DGF module that transfers structural cues from low to high resolution.
- Claimed novelty: first Laplacian-pyramid framework for UDC restoration; first application of DGF in UDC restoration.

### Number of Laplacian pyramid levels
- **Primary configuration: 3 levels** ("Taking a three-level pyramid as an example…", "Hence, we adopt the 3-level setting").
- Ablation studies also report **2 levels** and **4 levels**.

### Laplacian pyramid decomposition (classical, as described in Sec. 2.2)
- Given `I0 ∈ R^{h×w}`:
  - `I1 = downsample(I0)` → resolution `(h/2) × (w/2)`, produced by "applying a fixed-kernel weighted average to neighboring pixels".
  - Upsample `I1` back → `Î0`.
  - High-frequency residual: `h0 = I0 − Î0`.
- Process is repeated iteratively on `I1` to build a pyramid of high-frequency residuals plus a final low-frequency image.
- Invertibility is a classical LP property that the paper relies on conceptually.

### Low-frequency branch (Sec. 3.2)
- Role: global reconstruction; correct brightness attenuation and color distortion; preserve overall structure.
- Structure (explicitly stated):
  1. **Feature extraction module:** 3×3 convolution followed by **LeakyReLU** to extract shallow features.
  2. Then a **Dilated Residual Block (DRB)** "to capture multi-scale contextual information".
  3. Then an **encoder–decoder structure** for feature compression and recovery.
  4. Finally a **symmetric reconstruction module** that restores global color and illumination information in the low-frequency image.

### Deep Guided Filtering module (Sec. 3.3)
- Traditional Guided Filter formulation (Eqs. 2–5), cited as [32] (He et al. style guided filter):
  - Linear model in local window `ω_k`: `q_i = a_k G_i + b_k`, ∀i ∈ ω_k
  - Optimization: `min_{a_k,b_k} Σ_{i∈ω_k} ((a_k G_i + b_k − I_i)^2 + ε a_k^2)`
  - Regularization `ε` "typically set to 0.001"
  - Closed form: `a_k = cov_{ω_k}(G,I) / (σ_k^2 + ε)`, `b_k = μ_{I,k} − a_k μ_{G,k}`
  - Final output: `q_i = ā_i G_i + b̄_i`, where `ā_i`, `b̄_i` are averages of `a_k`, `b_k` over all windows containing pixel `i`
- **Why traditional GF is insufficient for UDC:**
  - The linear assumption is "overly idealized for complex UDC degradation scenarios".
  - Cannot effectively model nonlinear distortions such as screen interference, color shifts, and high-frequency artifacts.
  - Weighted guided filter [35] only partially addresses this.
- **What LPUDC's DGF changes:**
  1. **Replace local averaging with learnable weighted means.** Weights are dynamically learned by a neural network, adapting to image content.
  2. **Replace the fixed linear mapping (Eq. 5) with convolutional blocks**, removing the constraints of the fixed formula so output features can capture more expressive nonlinear relationships.
  3. **Role assignment (critical):**
     - Guidance image `G` = **input of the low-frequency restoration network**.
     - Input image `I` to the guided filter = **output of the low-frequency restoration network**.
  4. **Estimate transformation coefficients `a_k`, `b_k`** between input and guidance (instead of directly predicting the filtered image). Coefficients can be solved from `G` and `I`.
  5. Coefficients are **upsampled / aligned** to the spatial resolution of the **upper pyramid layer** to obtain `ā_i` and `b̄_i`.
  6. Filtered result obtained using the **standard guided-filter output formulation** (Eq. 5).
- Paper also states: a lightweight global pre-processing step first "mitigates minor artifacts and aligns structural layouts in the high-frequency image"; the HF network then processes the guided HF image.
- Cited DGF reference: **[9] H. Wu, S. Zheng, J. Zhang, K. Huang, "Fast end-to-end trainable guided filter", CVPR 2018.**
- Paper notes alternatives tried and found suboptimal: transferring only low-resolution inputs; concatenating input–output pairs.

### High-frequency branch (Sec. 3.4)
- Role: enhance high-frequency details (textures, edges, diffraction artifacts); compensate detail information that low-frequency restoration cannot recover.
- Pipeline (explicitly stated):
  1. **Inverse pixel shuffle** — compresses spatial information into the channel dimension; reduces computational overhead while retaining local features.
  2. **3×3 convolution** — adjusts channel number, extracts basic features.
  3. **DRB (residual blocks with dilated convolutions)** — expand receptive field, enhance perception of local and global information.
  4. **High-Resolution Network (HRNet)** — "multiple DRBs are connected in series"; continuously extracts and reinforces high-frequency details; focuses on correcting diffraction artifact areas.
  5. **Residual connections throughout HRNet** — ensure completeness of feature transmission, prevent gradient vanishing, improve training stability.
  6. **Final 3×3 convolution** — projects enhanced features back to the original channel number to **fuse with the high-frequency components of the Laplacian pyramid**.
- Conclusion section additionally names **"high-resolution residual blocks (HRB)"** in the high-frequency branch (equivalent in spirit to the DRB series inside the HRNet stage).

### Reconstruction process
- The network restores different pyramid levels and reconstructs the final high-resolution image progressively (pyramid reconstruction).
- HF branch output is fused with the Laplacian high-frequency components.
- The paper's loss (Eq. 7) compares the **enhanced image Laplacian pyramid** `Î_i` against the **ground-truth Laplacian pyramid** `I_i`, implying the model produces a restored image whose pyramid is supervised, and/or explicit predicted pyramid levels.

### Loss functions
- **Content loss (Eq. 6):** `L_con = Σ_{p=1}^{N} || Y_p − X_p ||` — L1 loss between predicted image `Y` and ground truth `X` (N = image size / pixel index).
- **Pyramid reconstruction loss (Eq. 7):**
  `L_pyr = Σ_{i=0}^{L} || Î_i − I_i ||_1`
  where `L` is the total number of pyramid levels, `Î_i` is the i-th level of the **enhanced image** Laplacian pyramid, `I_i` is the i-th level of the **ground-truth** Laplacian pyramid.
- **Total loss (Eq. 8):** `L = L_con + λ · L_pyr`
- **λ = 0.5** ("where λ is a weighting factor set to 0.5 in our experiments").

### Optimizer
- **Adam** [40]
- **β1 = 0.9, β2 = 0.999**

### Learning rate / scheduler
- **Initial LR = 2 × 10⁻⁴**
- **Final LR = 1 × 10⁻⁶**
- **Cosine Annealing** scheme [39]

### Batch size
- **Batch size = 1**

### Number of iterations
- **Total training iterations = 300K (300,000)**

### Data augmentation
- **Random horizontal flip**
- **Random vertical flip**
- Training data augmented with these; "For fair comparisons".

### Evaluation metrics
- (1) **PSNR** — higher is better
- (2) **SSIM** — higher is better
- (3) **LPIPS** [37] — lower is better
- (4) **DISTS** [38] — lower is better
- Same metrics as previous works for fair comparison.

### Dataset split
- **POLED** [1] and **TOLED** [1] (from Zhou et al., CVPR 2021).
- POLED: captured using an **RGBG PenTile** panel [36]; transmittance as low as **2.9%** → severe low-light, color distortion, diffraction artifacts.
- TOLED: captured using a **transparent OLED** panel; light transmittance approximately **20%** → relatively mild degradation.
- Each dataset: **300 images total = 240 train + 30 val + 30 test**.

### Image resolution
- **1024 × 2048** per image.
- Training uses **full-resolution images as input** (batch size 1).

### Hardware
- **Single NVIDIA RTX 3090 GPU**
- Framework: **PyTorch**

### Reported model complexity / speed (reference targets, NOT to be hardcoded)
| Dataset | Params (M) | Time (s) | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|---------|-----------:|---------:|------:|------:|-------:|-------:|
| POLED (LPUDC) | 4.0 | 0.07 | 33.99 | 0.9629 | 0.1678 | 0.1399 |
| TOLED (LPUDC) | 4.0 | 0.07 | 38.62 | 0.9805 | 0.0893 | 0.0858 |

### Ablation settings (explicitly in paper)

**A. Multi-scale fusion + guided filtering** (Tables 3 & 4)
| Multi-Scale Fusion | Guided Filtering | POLED PSNR | TOLED PSNR |
|:--:|:--:|--:|--:|
| ✗ | ✗ | 33.79 | 38.36 |
| ✓ | ✗ | 33.82 | 38.42 |
| ✓ | ✓ | 33.99 | 38.62 |

- Setting ✗/✗: "Each level of the pyramid processes independently without any cross-level connections."
- Setting ✓/✗: "Low-resolution features are upsampled and concatenated with high-resolution features directly." (paper body calls this "Direct feature concatenation"; user brief called it "multi-scale interaction without guided filtering")
- Setting ✓/✓: "Low-resolution features guide the enhancement of high-resolution features." (full LPUDC)

**B. Pyramid levels** (Tables 5 & 6)
| Levels | Params (M) | Time (s) | POLED PSNR | TOLED PSNR |
|-------:|-----------:|---------:|-----------:|------------:|
| 2 | 2.9 | 0.05 | 33.82 | 38.40 |
| 3 | 4.0 | 0.07 | 33.99 | 38.62 |
| 4 | 5.1 | 0.11 | 34.03 | 38.64 |

**C. Loss function** (Tables 7 & 8)
| L1 Loss | Laplacian Pyramid Loss | POLED PSNR | TOLED PSNR |
|:--:|:--:|--:|--:|
| ✓ | ✗ | 33.84 | 38.48 |
| ✓ | ✓ | 33.99 | 38.62 |

### Data availability statement in paper
- "The data that has been used is confidential."
- (Note: POLED/TOLED themselves originate from Zhou et al. CVPR 2021 and are widely redistributed in the UDC literature. This statement may refer to additional private captures. Our implementation must treat POLED/TOLED as external datasets the user must supply.)

---

## B. Details NOT Specified in the Paper

The following are **not given numerically or structurally** in the paper text. We do not invent paper values for them.

| Item | Status in paper |
|------|-----------------|
| Exact channel widths of low-frequency encoder–decoder | Not specified |
| Number of DRBs in low-frequency branch | Not specified |
| Number of encoder/decoder stages and skip-connection details | Not specified ("multi-scale encoder–decoder" only) |
| Exact architecture of "symmetric reconstruction module" | Not specified beyond "restores global color and illumination" |
| Dilated-convolution rates / spacing schedule inside DRB | Not specified |
| DRB internal layout (number of convs, residual scaling) | Not specified |
| Exact DGF neural architecture (layers, channels) | Not specified; paper cites Wu et al. CVPR 2018 [9] as the DGF module and describes conceptual modifications |
| Learnable-weight mean kernel size | Not specified |
| Convolutional-block layout that replaces GF linear mapping | Not specified |
| Inverse pixel shuffle factor | Not specified |
| Exact HRNet stage widths / how many DRBs "in series" | Not specified |
| Number of stages in high-frequency HRNet | Not specified |
| Laplacian downsample kernel (classical 5×5 binomial is standard, not stated) | Not specified |
| Upsampling kernel / interpolation mode for LP reconstruction | Not specified |
| Weight initialization | Not specified |
| Gradient clipping / gradient accumulation / AMP | Not specified (AMP etc. are engineering additions) |
| Validation frequency / checkpointing schedule | Not specified |
| Random seed | Not specified |
| Batch size for validation | Not specified |
| Whether input is normalized to [0,1] or [-1,1] | Not specified |
| LPIPS backbone (Alex/VGG/Squeeze) | Not specified |
| DISTS implementation version | Not specified |
| Exact cosine-annealing formula / T_max definition | Not specified (300K iterations implied as the schedule length) |
| Whether LR decays per-iteration or per-epoch | Implied per-iteration (iterations, not epochs, are the unit) |
| Use of EMA / Exponential Moving Average | Not specified |
| Mixed precision training | Not specified |
| Optimizer weight decay | Not specified (Adam default 0 assumed if not stated — but paper does not say) |
| Whether DRB uses pre-activation or post-activation | Not specified |
| LeakyReLU negative slope | Not specified (PyTorch default 0.01 used as engineering choice) |
| Exact definition of "global brightness/color restoration" beyond qualitative text | Not specified |

---

## C. Engineering Assumptions Made by Our Implementation

These are **our choices**, not paper claims. Each is flagged in code comments as `IMPLEMENTATION ASSUMPTION`.

### C.1 Laplacian pyramid
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| Downsample kernel | 5×5 binomial filter `[1,4,6,4,1]⊗[1,4,6,4,1]/256`, stride-2 convolution, pad to handle odd sizes | Classical Burt–Adelson LP; paper says "fixed-kernel weighted average" without giving the kernel |
| Upsample | Bilinear upsample + same 5×5 binomial blur (unsharp-style), then resize to exact target H×W | Matches classical LP reconstruction; avoids checkerboard from transposed conv |
| Odd spatial sizes | Crop/pad so residual `H0 = I0 − Î0` is exact | 1024×2048 is even, but custom datasets may not be |
| Number of levels `L` | Configurable; default 3 | Paper primary config is 3 |
| Pyramid loss indexing | Sum over all levels `i = 0 … L` of L1 between predicted-pyramid level and GT-pyramid level | Eq. 7 as written |

### C.2 Low-frequency branch
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| Shallow features | 3×3 Conv2d → LeakyReLU(0.01) | Paper explicitly says 3×3 conv + LeakyReLU; slope is PyTorch default (unspecified) |
| DRB count in LF stem | Configurable, default 2 | Paper says "employs a Dilated Residual Block"; number unspecified |
| DRB structure | Two 3×3 dilated convs + LeakyReLU + residual add; dilation rates (1, 2) or (1, 2, 4) configurable | Standard dilated residual design; paper does not give internals |
| Encoder–decoder | 3-level U-Net style with stride-2 down and bilinear up + 1×1 skip concat | Paper says "multi-scale encoder–decoder for feature compression and recovery"; widths unspecified |
| Channel schedule (LF enc) | [32, 64, 128] | Engineering choice to stay near 4.0M total params |
| Symmetric reconstruction | Mirror of encoder: 1×1 + 3×3 convs ending in 3×3 conv to `out_ch` (RGB residual or direct image) | Paper: "symmetric reconstruction module restores global color and illumination" |
| Output form | LF branch predicts a residual added to the LP low-frequency image (global restoration), **and** also produces a guidance feature map for DGF | Residual learning is standard for restoration; paper does not specify residual vs direct |

### C.3 Deep Guided Filtering
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| Module lineage | Implementation follows the *conceptual* modifications described in Sec. 3.3, structured on Wu et al. CVPR 2018 "Fast end-to-end trainable guided filter" (paper ref [9]) | Paper names [9] as the DGF module it adopts |
| Guidance `G` | Input of low-frequency restoration network (the LF LP component / shallow features derived from it) | Explicitly stated in Sec. 3.3 |
| Filter input `I` | Output of low-frequency restoration network | Explicitly stated in Sec. 3.3 |
| Learnable weighted mean | Depthwise conv / grouped conv with learned spatial weights over a `r×r` neighborhood (default r=2 or 4), plus normalization | "weights are dynamically learned by a neural network" |
| Learnable linear mapping | Small conv block (e.g. 3×3 conv → ReLU → 3×3 conv) replacing `q = ā G + b̄` | "replace this linear mapping with convolutional blocks" |
| Coefficient estimation | Predict `a` and `b` maps from `(G, I)` via the learnable weighted-mean guided-filter closed form, then refine with convs | Paper: "shifting focus from directly predicting the filtered image to estimating the linear transformation coefficients a_k and b_k" |
| Coefficient upsampling | Bilinear upsample `a`, `b` to the spatial size of the upper pyramid level | Explicitly stated: "coefficients are then upsampled … aligned with the spatial resolution of the upper pyramid layer" |
| Final fusion | `q = conv_block(ā_up · G_up + b̄_up)` — standard GF output form, but with conv refinement after (documented deviation because paper also says linear mapping is replaced by convs) | Reconciles Eq. 5 usage with the "replace linear mapping with conv blocks" statement |
| ε | 0.001 | Paper: "typically set to 0.001" |
| Differentiability | Fully differentiable; no `detach()` on guidance | End-to-end trainable as required |

### C.4 High-frequency branch
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| Inverse pixel shuffle factor | Default 2 (pixel-unshuffle) | Unspecified; factor 2 halves spatial dims and ×4 channels, standard |
| 3×3 conv after unshuffle | Present; maps `C*factor² → base_ch` | Paper explicitly |
| DRB stack before HRNet | Configurable, default 2 | Paper mentions DRBs then HRNet of serial DRBs |
| "HRNet" interpretation | **Not** the full HRNet (Wang et al.) multi-branch architecture. Implemented as a **high-resolution residual trunk**: a series of DRBs with global residual connection, operating at the (reduced) resolution after inverse pixel shuffle | Paper text: "a High-Resolution Network (HRNet) is introduced, where multiple DRBs are connected in series" + "relies on residual connections throughout". Conclusion calls the blocks HRB. Full HRNet is not described and would break the 4.0M budget |
| HRB/DRB count in HRNet trunk | Configurable, default 4 | Unspecified |
| Final projection | 3×3 conv back to original channel count, then residual add with the Laplacian HF component | Paper explicitly: "projected back to the original channel number through a 3×3 convolution to fuse with the high-frequency components of the Laplacian pyramid" |
| Fusion form | `H_restored = H_Laplacian + projected_features` (residual fusion) | "fuse with" + residual philosophy of the paper; exact add vs concat unspecified |

### C.5 Model integration
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| Overall flow | Decompose degraded image → process LF at coarsest level → propagate guidance upward via DGF → enhance HF at each finer level → reconstruct image from predicted pyramid | Matches Fig. 2 / Sec. 3 narrative |
| LF branch applied | At the **coarsest** LP level (after `L` downsamples) | Paper: "The bottom layer is processed using an encoder–decoder architecture" |
| HF branch applied | At each of the `L` high-frequency levels, guided by DGF from the level below | Paper Sec. 3.4 + Fig. 2/3 |
| DGF connection | Between consecutive pyramid levels (coarse → fine) | "Each level is connected by a Deep Guided Filtering (DGF) module" |
| Output | Final restored image at full resolution + list of predicted pyramid levels (for `L_pyr`) | Required by Eq. 7 |
| Parameter budget target | ~4.0M | Paper reports 4.0M for 3-level model; we tune channel widths to approach this and **report our actual measured count** |

### C.6 Training / evaluation engineering (not in paper)
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| Pixel range | Inputs/targets normalized to **[0, 1]** float RGB | Common; paper unspecified |
| Content loss reduction | Mean L1 (PyTorch `L1Loss(reduction='mean')`) instead of sum over all pixels | Eq. 6 writes a sum over N pixels; with 1024×2048×3 a raw sum would dominate and be scale-fragile. Mean is the standard practical equivalent. **Documented deviation.** |
| Pyramid loss reduction | Mean L1 per level, summed over levels | Same scale-consistency reasoning as above |
| Cosine schedule | Per-iteration cosine annealing from `lr_init` to `lr_min` over `total_iters`, using `CosineAnnealingLR` with `T_max=total_iters` and `eta_min=lr_min`, stepped every iteration | Paper gives endpoints + scheme name only |
| AMP | Optional (`--amp`), default **off** for reproduction fidelity | Paper does not mention AMP; reproduction config keeps it off |
| Gradient accumulation | Optional, default 1 (no accumulation) | Engineering convenience |
| Gradient checkpointing | Optional, default off | Memory safety for low-VRAM GPUs |
| Val metric | PSNR/SSIM on full images (or center-crop if OOM), LPIPS/DISTS optional | Paper does not describe val protocol |
| Seed | Default 42, stored in checkpoints/config | Reproducibility requirement of the user brief |
| LPIPS backbone | `alex` if `lpips` package available; otherwise skipped with a clear warning | Paper does not specify backbone |
| DISTS | Official `dists-pytorch` implementation with pretrained weights | Metric returns `null` if the package or weights are unavailable |
| Hardware print | Always print Device/GPU/VRAM; if CUDA unavailable, continue on CPU with a warning | User requirement |

### C.7 Dataset / configs
| Choice | Assumption | Rationale |
|--------|------------|-----------|
| On-disk layout | `root/{train,val,test}/{degraded,gt}/` with filename matching | User requirement; POLED/TOLED official release folders may differ — a converter note is provided |
| Pair matching | Match by stem (filename without extension); strict mode fails on missing pairs | User requirement |
| Augmentation | Random H flip (p=0.5), random V flip (p=0.5), applied jointly to degraded+GT; none for val/test | Paper |
| DEBUG config | Separate `configs/lpudc_poled_debug.yaml` with smaller crop/resize and fewer iters | User requirement: never silently change the reproduction config |

---

## D. Reproduction Targets (Reference Only — Never Hardcoded in Code)

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | Params | Time |
|---------|------:|------:|-------:|-------:|-------:|-----:|
| POLED | 33.99 | 0.9629 | 0.1678 | 0.1399 | 4.0M | 0.07s |
| TOLED | 38.62 | 0.9805 | 0.0893 | 0.0858 | 4.0M | 0.07s |

These numbers appear **only** in this document and in `docs/REPRODUCTION_REPORT.md` as *paper-reported* values. Evaluation code never reads them.

---

## E. Paper Result Tables (for the Reproduction Report)

### POLED comparison (Table 1)
| Method | Params (M) | Time (s) | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|--------|-----------:|---------:|------:|------:|-------:|-------:|
| MSUNET | 8.9 | 0.08 | 29.17 | 0.9393 | 0.2239 | 0.1746 |
| DAGF | 1.1 | 1.12 | 32.29 | 0.9509 | 0.2163 | 0.1913 |
| PDCRN | 4.7 | 0.08 | 32.99 | 0.9578 | 0.2102 | 0.2075 |
| ResUNet | 16.5 | 0.43 | 31.39 | 0.9506 | 0.2129 | 0.1823 |
| UDC-UNet | 5.7 | 0.30 | 33.12 | 0.9592 | 0.1924 | 0.1617 |
| BNUDC | 4.6 | 0.08 | 33.39 | 0.9610 | 0.1748 | 0.1511 |
| FSI | 3.8 | 0.07 | 33.85 | 0.9627 | 0.1738 | 0.1457 |
| SGSFormer | 9.3 | 0.26 | 33.28 | 0.9627 | 0.1573 | 0.1286 |
| **LPUDC** | **4.0** | **0.07** | **33.99** | **0.9629** | **0.1678** | **0.1399** |

### TOLED comparison (Table 2)
| Method | Params (M) | Time (s) | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|--------|-----------:|---------:|------:|------:|-------:|-------:|
| MSUNET | 8.9 | 0.08 | 37.40 | 0.9756 | 0.1093 | 0.1052 |
| DAGF | 1.1 | 1.12 | 36.49 | 0.9716 | 0.1392 | 0.1217 |
| IPIUer | 24.7 | 0.10 | 38.18 | 0.9796 | 0.1128 | 0.1050 |
| BAIDU | 20.0 | 0.18 | 38.23 | 0.9799 | 0.1026 | 0.0966 |
| UDC-UNet | 5.7 | 0.30 | 38.10 | 0.9796 | 0.1003 | 0.0971 |
| BNUDC | 4.6 | 0.08 | 38.22 | 0.9798 | 0.0988 | 0.0964 |
| FSI | 3.8 | 0.07 | 38.60 | 0.9805 | 0.0979 | 0.0915 |
| SGSFormer | 9.3 | 0.26 | 38.42 | 0.9806 | 0.0926 | 0.0883 |
| **LPUDC** | **4.0** | **0.07** | **38.62** | **0.9805** | **0.0893** | **0.0858** |
