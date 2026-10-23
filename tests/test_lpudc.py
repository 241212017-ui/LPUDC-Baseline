"""Automated tests for LPUDC.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transforms import pil_to_tensor
from losses.lpudc_loss import LPUDCLoss
from metrics.psnr import psnr
from metrics.ssim import ssim
from models.blocks import DilatedResidualBlock
from models.dgf import DeepGuidedFilter
from models.high_frequency import HighFrequencyBranch
from models.laplacian_pyramid import LaplacianPyramid
from models.low_frequency import LowFrequencyBranch
from models.lpudc import LPUDC, LPUDCConfig
from models.reconstruction import PyramidReconstructor
from utils import count_parameters, load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def tiny_config() -> LPUDCConfig:
    return LPUDCConfig(
        pyramid_levels=3,
        in_channels=3,
        lf_base_channels=16,
        lf_encoder_channels=(16, 32, 64),
        lf_stem_drb_count=1,
        lf_guidance_channels=32,
        dgf_coeff_channels=16,
        hf_base_channels=16,
        hf_pre_drb_count=1,
        hf_hrnet_drb_count=2,
        debug_shapes=True,
    )


# ---------------------------------------------------------------------------
# Laplacian pyramid
# ---------------------------------------------------------------------------

def test_laplacian_pyramid_dimensions():
    lp = LaplacianPyramid(3)
    x = torch.randn(1, 3, 64, 64)
    residuals, coarse, levels = lp(x)
    assert len(residuals) == 3
    assert len(levels) == 4
    assert residuals[0].shape == (1, 3, 64, 64)
    assert residuals[1].shape == (1, 3, 32, 32)
    assert residuals[2].shape == (1, 3, 16, 16)
    assert coarse.shape == (1, 3, 8, 8)


def test_laplacian_pyramid_reconstruction_near_identity():
    lp = LaplacianPyramid(3)
    x = torch.rand(1, 3, 64, 64)
    residuals, coarse, _ = lp(x)
    recon = lp.reconstruct(residuals, coarse)
    assert recon.shape == x.shape
    # Bilinear expand is not a perfect inverse of binomial downsample, but
    # reconstruction error should be small relative to signal range.
    mae = (recon - x).abs().mean().item()
    assert mae < 0.05, f"reconstruction MAE too high: {mae}"


def test_laplacian_pyramid_grad():
    lp = LaplacianPyramid(2)
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    residuals, coarse, _ = lp(x)
    loss = sum(r.abs().mean() for r in residuals) + coarse.abs().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# DRB
# ---------------------------------------------------------------------------

def test_drb_forward():
    drb = DilatedResidualBlock(16, dilations=(1, 2, 4))
    x = torch.randn(2, 16, 32, 32)
    y = drb(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_drb_grad():
    drb = DilatedResidualBlock(8)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    drb(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Low-frequency branch
# ---------------------------------------------------------------------------

def test_low_frequency_branch():
    lf = LowFrequencyBranch(
        in_channels=3,
        base_channels=16,
        encoder_channels=(16, 32, 64),
        stem_drb_count=1,
        guidance_channels=32,
    )
    x = torch.randn(1, 3, 32, 32)
    restored, guidance = lf(x)
    assert restored.shape == x.shape
    assert guidance.shape[0] == 1
    assert guidance.shape[1] == 32
    assert guidance.shape[-2:] == x.shape[-2:]
    assert torch.isfinite(restored).all()


# ---------------------------------------------------------------------------
# DGF
# ---------------------------------------------------------------------------

def test_dgf_forward_shapes():
    dgf = DeepGuidedFilter(
        guidance_channels=32,
        input_channels=16,
        coefficient_channels=16,
    )
    g = torch.randn(1, 32, 16, 16)
    i_feat = torch.randn(1, 16, 32, 32)  # upper pyramid level (finer)
    out = dgf(g, i_feat, target_size=i_feat.shape[-2:])
    assert out.shape == i_feat.shape
    assert torch.isfinite(out).all()


def test_dgf_gradients():
    dgf = DeepGuidedFilter(guidance_channels=8, input_channels=8, coefficient_channels=8)
    g = torch.randn(1, 8, 16, 16, requires_grad=True)
    i_feat = torch.randn(1, 8, 32, 32, requires_grad=True)
    out = dgf(g, i_feat)
    out.mean().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    assert i_feat.grad is not None and torch.isfinite(i_feat.grad).all()
    # Guidance must not be detached — gradients should flow to G.
    assert g.grad.abs().sum() > 0


def test_dgf_constant_inputs_remain_finite():
    dgf = DeepGuidedFilter(guidance_channels=3, input_channels=3, coefficient_channels=3)
    guidance = torch.ones(1, 3, 8, 8, requires_grad=True)
    image = torch.zeros(1, 3, 8, 8, requires_grad=True)

    output = dgf(guidance, image)
    output.mean().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(guidance.grad).all()
    assert torch.isfinite(image.grad).all()


def test_dgf_coefficient_upsample_alignment():
    dgf = DeepGuidedFilter(guidance_channels=4, input_channels=4, coefficient_channels=4)
    g = torch.randn(1, 4, 8, 8)
    i_feat = torch.randn(1, 4, 16, 16)
    out = dgf(g, i_feat, target_size=(16, 16))
    assert out.shape[-2:] == (16, 16)


# ---------------------------------------------------------------------------
# High-frequency branch
# ---------------------------------------------------------------------------

def test_high_frequency_branch():
    hf = HighFrequencyBranch(
        in_channels=3,
        base_channels=16,
        unshuffle_factor=2,
        pre_drb_count=1,
        hrnet_drb_count=2,
        guidance_channels=16,
    )
    res = torch.randn(1, 3, 64, 64)
    guided = torch.randn(1, 16, 64, 64)
    out = hf(res, guided_feat=guided)
    assert out.shape == res.shape
    assert torch.isfinite(out).all()


def test_high_frequency_branch_without_guidance():
    hf = HighFrequencyBranch(
        in_channels=3,
        base_channels=16,
        pre_drb_count=1,
        hrnet_drb_count=2,
        guidance_channels=0,
    )
    res = torch.randn(1, 3, 32, 32)
    out = hf(res, guided_feat=None)
    assert out.shape == res.shape


def test_high_frequency_confidence_gate_gradients():
    hf = HighFrequencyBranch(
        in_channels=3,
        base_channels=16,
        pre_drb_count=1,
        hrnet_drb_count=2,
        guidance_channels=16,
        confidence_gate=True,
    )
    residual = torch.randn(1, 3, 32, 32, requires_grad=True)
    guidance = torch.randn(1, 16, 32, 32, requires_grad=True)

    output = hf(residual, guided_feat=guidance)
    output.mean().backward()

    assert output.shape == residual.shape
    assert torch.isfinite(output).all()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert guidance.grad is not None and torch.isfinite(guidance.grad).all()
    assert hf.gate_conv.weight.grad is not None


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def test_pyramid_reconstructor():
    rec = PyramidReconstructor(3)
    residuals = [torch.randn(1, 3, 64, 64), torch.randn(1, 3, 32, 32), torch.randn(1, 3, 16, 16)]
    coarse = torch.randn(1, 3, 8, 8)
    restored, levels = rec(residuals, coarse)
    assert restored.shape == (1, 3, 64, 64)
    assert len(levels) == 4
    PyramidReconstructor.assert_shapes(restored, levels, (64, 64), 3)


# ---------------------------------------------------------------------------
# Full LPUDC
# ---------------------------------------------------------------------------

def test_lpudc_forward(tiny_config):
    model = LPUDC(tiny_config)
    x = torch.rand(1, 3, 64, 64)
    out = model(x, return_extras=True)
    assert out["restored"].shape == x.shape
    assert len(out["pred_levels"]) == tiny_config.pyramid_levels + 1
    assert len(out["degraded_levels"]) == tiny_config.pyramid_levels + 1
    assert out["lf_restored"].shape[-2:] == out["pred_levels"][-1].shape[-2:]
    assert torch.isfinite(out["restored"]).all()


def test_lpudc_confidence_gate_forward(tiny_config):
    config = LPUDCConfig.from_dict(
        {**tiny_config.to_dict(), "confidence_gate": True}
    )
    model = LPUDC(config)
    image = torch.rand(1, 3, 64, 64)
    output = model(image, return_extras=False)

    assert output["restored"].shape == image.shape
    assert torch.isfinite(output["restored"]).all()


def test_lpudc_identity_preserves_pyramid_residuals(tiny_config):
    model = LPUDC(tiny_config).eval()
    with torch.no_grad():
        for parameter in model.low_frequency.recon_head.parameters():
            parameter.zero_()
        for branch in model.high_frequency:
            branch.out_conv.weight.zero_()
            branch.out_conv.bias.zero_()
            branch.fusion_scale.zero_()

        image = torch.rand(1, 3, 64, 64)
        output = model(image, return_extras=False)
        _, _, input_levels = model.pyramid(image)

    for predicted, original in zip(output["pred_levels"], input_levels):
        assert torch.allclose(predicted, original, atol=1e-6)


def test_lpudc_dgf_uses_low_frequency_input_and_output(tiny_config):
    model = LPUDC(tiny_config).eval()
    captured = {}

    def capture_low_frequency(_module, inputs, output):
        captured["lf_input"] = inputs[0].detach()
        captured["lf_output"] = output[0].detach()

    def capture_dgf(_module, inputs):
        captured["dgf_guidance"] = inputs[0].detach()
        captured["dgf_input"] = inputs[1].detach()

    low_frequency_hook = model.low_frequency.register_forward_hook(capture_low_frequency)
    dgf_hook = model.dgf_modules[0].register_forward_pre_hook(capture_dgf)
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        model(image, return_extras=False)
        _, _, input_levels = model.pyramid(image)
    low_frequency_hook.remove()
    dgf_hook.remove()

    assert torch.allclose(captured["dgf_guidance"], captured["lf_input"])
    assert torch.allclose(captured["dgf_input"], captured["lf_output"])
    assert torch.allclose(captured["lf_input"], input_levels[-1])


def test_lpudc_param_count_reasonable(tiny_config):
    model = LPUDC(tiny_config)
    total, trainable = count_parameters(model)
    assert total > 10_000
    assert trainable == total


def test_lpudc_fusion_modes(tiny_config):
    for mode in (0, 1, 2):
        cfg = LPUDCConfig.from_dict({**tiny_config.to_dict(), "fusion_mode": mode})
        model = LPUDC(cfg)
        x = torch.rand(1, 3, 64, 64)
        out = model(x, return_extras=False)
        assert out["restored"].shape == x.shape


def test_lpudc_levels_2_and_4(tiny_config):
    for levels in (2, 4):
        cfg = LPUDCConfig.from_dict({**tiny_config.to_dict(), "pyramid_levels": levels})
        model = LPUDC(cfg)
        x = torch.rand(1, 3, 64, 64)
        out = model(x, return_extras=False)
        assert out["restored"].shape == x.shape
        assert len(out["pred_levels"]) == levels + 1


def test_lpudc_odd_input_size():
    cfg = LPUDCConfig(
        pyramid_levels=3,
        lf_base_channels=16,
        lf_encoder_channels=(16, 32),
        lf_stem_drb_count=1,
        lf_guidance_channels=32,
        dgf_coeff_channels=16,
        hf_base_channels=16,
        hf_pre_drb_count=1,
        hf_hrnet_drb_count=2,
        debug_shapes=True,
    )
    model = LPUDC(cfg)
    x = torch.rand(1, 3, 63, 61)
    out = model(x, return_extras=False)
    assert out["restored"].shape[0] == 1
    assert out["restored"].shape[1] == 3
    # Restored should be close to input spatial size (pad/crop tolerance).
    assert abs(out["restored"].shape[-2] - 63) <= 2
    assert abs(out["restored"].shape[-1] - 61) <= 2


def test_lpudc_grad_end_to_end(tiny_config):
    model = LPUDC(tiny_config)
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    out = model(x, return_extras=False)
    loss = out["restored"].abs().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Every parameter should receive a gradient (or at least some do).
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_loss_components():
    loss_fn = LPUDCLoss(pyramid_weight=0.5, use_pyramid_loss=True, num_levels=3)
    pred = torch.rand(1, 3, 32, 32)
    gt = torch.rand(1, 3, 32, 32)
    # Build predicted levels from pred via the same pyramid
    lp = LaplacianPyramid(3)
    _, _, pred_levels = lp(pred)
    losses = loss_fn(pred, gt, pred_levels)
    assert set(losses.keys()) == {"total_loss", "content_loss", "pyramid_loss"}
    assert torch.isfinite(losses["total_loss"])
    expected = losses["content_loss"] + 0.5 * losses["pyramid_loss"]
    assert torch.allclose(losses["total_loss"], expected, atol=1e-6)


def test_loss_l1_only_ablation():
    loss_fn = LPUDCLoss(use_pyramid_loss=False, num_levels=3)
    pred = torch.rand(1, 3, 16, 16)
    gt = torch.rand(1, 3, 16, 16)
    losses = loss_fn(pred, gt, None)
    assert losses["pyramid_loss"].item() == 0.0
    assert torch.allclose(losses["total_loss"], losses["content_loss"])


def test_loss_grad():
    loss_fn = LPUDCLoss(pyramid_weight=0.5, num_levels=2)
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    gt = torch.rand(1, 3, 32, 32)
    lp = LaplacianPyramid(2)
    _, _, levels = lp(pred)
    losses = loss_fn(pred, gt, levels)
    losses["total_loss"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_psnr_ssim_identity():
    x = torch.rand(1, 3, 64, 64)
    p = psnr(x, x)
    s = ssim(x, x)
    assert p.item() > 60  # essentially identical
    assert s.item() > 0.99


def test_psnr_decreases_with_noise():
    x = torch.rand(1, 3, 64, 64)
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    assert psnr(x, x).item() > psnr(noisy, x).item()


# ---------------------------------------------------------------------------
# Dataset pairing
# ---------------------------------------------------------------------------

def test_dataset_pairing(tmp_path):
    from PIL import Image

    from data.dataset import PairedImageDataset
    from data.transforms import PairedCompose, PairedRandomFlip

    root = tmp_path / "ds"
    for split in ("train", "val"):
        for sub in ("degraded", "gt"):
            (root / split / sub).mkdir(parents=True)
    # Create 3 pairs for train, 2 for val
    for i in range(3):
        deg = Image.new("RGB", (32, 32), color=(i * 40, 10, 10))
        gt = Image.new("RGB", (32, 32), color=(i * 40, 200, 200))
        deg.save(root / "train" / "degraded" / f"img_{i:03d}.png")
        gt.save(root / "train" / "gt" / f"img_{i:03d}.png")
    for i in range(2):
        deg = Image.new("RGB", (32, 32), color=(10, i * 40, 10))
        gt = Image.new("RGB", (32, 32), color=(200, i * 40, 200))
        deg.save(root / "val" / "degraded" / f"v_{i}.png")
        gt.save(root / "val" / "gt" / f"v_{i}.png")

    tf = PairedCompose([PairedRandomFlip(0.5, 0.5)])
    train_ds = PairedImageDataset(root, split="train", transform=tf)
    val_ds = PairedImageDataset(root, split="val", transform=None)
    assert len(train_ds) == 3
    assert len(val_ds) == 2
    sample = train_ds[0]
    assert sample["degraded"].shape[0] == 3
    assert sample["gt"].shape == sample["degraded"].shape
    assert "name" in sample

    # Missing pair detection
    (root / "train" / "degraded" / "orphan.png").write_bytes(b"not-an-image")
    with pytest.raises(Exception):
        PairedImageDataset(root, split="train", strict=True)


def test_transforms_deterministic_eval():
    from data.transforms import build_eval_transforms

    import torch

    tf = build_eval_transforms(crop=16)
    x = torch.rand(3, 32, 32)
    y = torch.rand(3, 32, 32)
    x1, y1 = tf(x.clone(), y.clone())
    x2, y2 = tf(x.clone(), y.clone())
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert x1.shape[-2:] == (16, 16)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(tmp_path, tiny_config):
    model = LPUDC(tiny_config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    path = tmp_path / "ckpt.pth"
    save_checkpoint(
        path,
        model,
        opt,
        sch,
        iteration=7,
        best_metrics={"best_psnr": 30.1},
        config={"model": tiny_config.to_dict()},
    )
    model2 = LPUDC(tiny_config)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    payload = load_checkpoint(path, model2, opt2, sch2)
    assert payload["iteration"] == 7
    assert payload["best_metrics"]["best_psnr"] == 30.1
    # State dicts match
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


# ---------------------------------------------------------------------------
# Overfitting sanity check (tiny synthetic data)
# ---------------------------------------------------------------------------

def test_lpudc_can_overfit_tiny_pair(tiny_config):
    """The model should reduce loss on a single fixed pair if implemented correctly."""
    torch.manual_seed(0)
    model = LPUDC(tiny_config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = LPUDCLoss(pyramid_weight=0.5, num_levels=3)

    target = torch.rand(1, 3, 64, 64)
    degraded = (target * 0.5 + 0.1).clamp(0, 1)  # simple synthetic degradation

    losses = []
    for i in range(80):
        opt.zero_grad()
        out = model(degraded, return_extras=False)
        losses_d = loss_fn(out["restored"], target, out.get("pred_levels"))
        loss = losses_d["total_loss"]
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # The corrected residual path needs more than a few dozen updates to fit.
    assert losses[-1] < losses[0] * 0.5, (
        f"Overfit failed: first={losses[0]:.4f} last={losses[-1]:.4f}"
    )
    baseline_psnr = psnr(degraded, target).item()
    with torch.no_grad():
        pred = model(degraded, return_extras=False)["restored"].clamp(0, 1)
        final_psnr = psnr(pred, target).item()
    assert final_psnr > baseline_psnr + 1.0, (
        f"PSNR did not improve over input: {baseline_psnr:.2f} -> {final_psnr:.2f}"
    )
