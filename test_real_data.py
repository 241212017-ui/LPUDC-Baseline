import sys
sys.path.insert(0, r'C:\Users\Razzak\OneDrive\Desktop\LPUDC')
import yaml
from data import build_dataloader_real

with open(r'C:\Users\Razzak\OneDrive\Desktop\LPUDC\configs\lpudc_poled_real.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['training']['batch_size'] = 1
cfg['training']['num_workers'] = 0

loader = build_dataloader_real(cfg, split='train', train=True, batch_size=1, num_workers=0)
print(f'Train loader: {len(loader)} batches')

batch = next(iter(loader))
print(f'Batch degraded: {batch["degraded"].shape}, gt: {batch["gt"].shape}')
print(f'Names: {batch["name"]}')
print(f'Range degraded: [{batch["degraded"].min():.4f}, {batch["degraded"].max():.4f}]')
print(f'Range gt: [{batch["gt"].min():.4f}, {batch["gt"].max():.4f}]')