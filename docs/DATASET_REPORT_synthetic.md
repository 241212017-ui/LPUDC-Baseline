# Dataset Report (LPUDC)

- **Root:** `data\synthetic_tiny`

## Split: `train`

- **Degraded images:** 8
- **GT images:** 8
- **Paired:** 8
- **Missing GT:** 0
- **Missing degraded:** 0
- **Corrupted:** 0
- **Dimension mismatches:** 0
- **Duplicate degraded (hash):** 0
- **Duplicate GT (hash):** 0
- **Resolutions (top):** [('64x64', 8)]
- **Aspect ratios (top):** [('1.000', 8)]
- **Channels:** {'RGB(3)': 8}

### Pixel statistics (sampled)

| | min | max | mean | std | brightness mean |
|---|---:|---:|---:|---:|---:|
| Degraded | 0.0000 | 0.7490 | 0.3454 | 0.1242 | 0.3454 |
| GT | 0.0118 | 1.0000 | 0.5828 | 0.2045 | 0.5828 |

### GT − Degraded difference statistics

- mean: 0.23733575642108917
- mean abs: 0.23798513412475586
- std: 0.11406173557043076
- p95 abs: 0.43529412150382996
- p99 abs: 0.5137255191802979

## Split: `val`

- **Degraded images:** 2
- **GT images:** 2
- **Paired:** 2
- **Missing GT:** 0
- **Missing degraded:** 0
- **Corrupted:** 0
- **Dimension mismatches:** 0
- **Duplicate degraded (hash):** 0
- **Duplicate GT (hash):** 0
- **Resolutions (top):** [('64x64', 2)]
- **Aspect ratios (top):** [('1.000', 2)]
- **Channels:** {'RGB(3)': 2}

### Pixel statistics (sampled)

| | min | max | mean | std | brightness mean |
|---|---:|---:|---:|---:|---:|
| Degraded | 0.0000 | 0.5725 | 0.2966 | 0.1161 | 0.2966 |
| GT | 0.0275 | 1.0000 | 0.5890 | 0.2073 | 0.5890 |

### GT − Degraded difference statistics

- mean: 0.29242369532585144
- mean abs: 0.29259058833122253
- std: 0.12060631066560745
- p95 abs: 0.4901960790157318
- p99 abs: 0.5568627119064331

## Split: `test`

- **Degraded images:** 2
- **GT images:** 2
- **Paired:** 2
- **Missing GT:** 0
- **Missing degraded:** 0
- **Corrupted:** 0
- **Dimension mismatches:** 0
- **Duplicate degraded (hash):** 0
- **Duplicate GT (hash):** 0
- **Resolutions (top):** [('64x64', 2)]
- **Aspect ratios (top):** [('1.000', 2)]
- **Channels:** {'RGB(3)': 2}

### Pixel statistics (sampled)

| | min | max | mean | std | brightness mean |
|---|---:|---:|---:|---:|---:|
| Degraded | 0.0000 | 0.7294 | 0.3296 | 0.1357 | 0.3296 |
| GT | 0.0118 | 1.0000 | 0.5711 | 0.2123 | 0.5711 |

### GT − Degraded difference statistics

- mean: 0.24151380360126495
- mean abs: 0.24192006886005402
- std: 0.11497388780117035
- p95 abs: 0.43529412150382996
- p99 abs: 0.5254901647567749

## Visualization grid

![grid](../results/dataset_analysis/viz_grid.png)

Each cell: [DEGRADED | GROUND TRUTH]

## Compatibility with LPUDC

All requested splits contain paired degraded/GT images. You may proceed to training with `scripts/train_custom.py` or `train.py --config configs/custom_dataset.yaml`.
