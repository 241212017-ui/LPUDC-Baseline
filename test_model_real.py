import sys
sys.path.insert(0, r'C:\Users\Razzak\OneDrive\Desktop\LPUDC')
import yaml
import torch
from models.lpudc import LPUDCConfig, build_lpudc
from data import build_dataloader_real

with open(r'C:\Users\Razzak\OneDrive\Desktop\LPUDC\configs\lpudc_poled_real.yaml') as f:
    cfg = yaml.safe_load(f)

# Use tiny model for memory test
model_cfg = cfg.get('model', {})
model_cfg['lf_base_channels'] = 16
model_cfg['lf_encoder_channels'] = [16, 32, 64]
model_cfg['lf_stem_drb_count'] = 1
model_cfg['lf_guidance_channels'] = 32
model_cfg['dgf_coeff_channels'] = 16
model_cfg['hf_base_channels'] = 16
model_cfg['hf_pre_drb_count'] = 1
model_cfg['hf_hrnet_drb_count'] = 2

model = build_lpudc(model_cfg)
print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# Test with real data batch
cfg['training']['batch_size'] = 1
cfg['training']['num_workers'] = 0
loader = build_dataloader_real(cfg, split='train', train=True, batch_size=1, num_workers=0)
batch = next(iter(loader))
x = batch['degraded']
print(f'Input shape: {x.shape}')

with torch.no_grad():
    out = model(x, return_extras=True)
print(f'Output shape: {out["restored"].shape}')
print('Forward pass OK!')