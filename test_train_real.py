import sys
sys.path.insert(0, r'C:\Users\Razzak\OneDrive\Desktop\LPUDC')
import yaml
import torch
import torch.nn as nn
from models.lpudc import build_lpudc
from losses.lpudc_loss import build_loss
from data import build_dataloader_real
from utils import set_seed

with open(r'C:\Users\Razzak\OneDrive\Desktop\LPUDC\configs\lpudc_poled_real.yaml') as f:
    cfg = yaml.safe_load(f)

# Use tiny model for quick test
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

# Data
cfg['training']['batch_size'] = 1
cfg['training']['num_workers'] = 0
train_loader = build_dataloader_real(cfg, split='train', train=True, batch_size=1, num_workers=0)

# Loss
loss_fn = build_loss({
    **cfg.get('loss', {}),
    'pyramid_levels': model_cfg.get('pyramid_levels', 3),
})

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

# Training loop (3 iterations)
model.train()
set_seed(42)

for i, batch in enumerate(train_loader):
    if i >= 3:
        break
    x = batch['degraded']
    gt = batch['gt']

    optimizer.zero_grad()
    out = model(x)
    loss_dict = loss_fn(out['restored'], gt)
    loss = loss_dict['total_loss']
    loss.backward()
    optimizer.step()

    print(f'Iter {i+1}: total={loss_dict["total_loss"].item():.4f}, '
          f'content={loss_dict["content_loss"].item():.4f}, '
          f'pyramid={loss_dict["pyramid_loss"].item():.4f}')

print('Quick training test OK!')