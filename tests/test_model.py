import torch

from streetshield.config import ModelConfig
from streetshield.modeling import build_detector, load_detector, save_checkpoint
from streetshield.training import combine_losses


def _tiny_model():
    cfg = ModelConfig(pretrained=False, fixed_size=[160, 128])
    return cfg, build_detector(cfg, num_classes=3)


def test_train_step_returns_combined_loss():
    _, model = _tiny_model()
    model.train()
    images = [torch.rand(3, 120, 160)]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 60.0]]), "labels": torch.tensor([1])}]
    total, parts = combine_losses(model(images, targets), lambda_reg=2.0)
    assert torch.isfinite(total)
    assert abs(parts["total"] - (parts["l_cls"] + 2.0 * parts["l_reg"])) < 1e-4
    total.backward()


def test_checkpoint_roundtrip(tmp_path):
    cfg, model = _tiny_model()
    save_checkpoint(model, tmp_path / "m.pt", cfg, ["face", "plate"], epoch=3)
    loaded, classes, ckpt = load_detector(tmp_path / "m.pt", torch.device("cpu"))
    assert classes == ["face", "plate"] and ckpt["epoch"] == 3
    with torch.no_grad():
        out = loaded([torch.rand(3, 120, 160)])[0]
    assert set(out) == {"boxes", "scores", "labels"}


def test_fixed_size_maps_boxes_back_to_original_resolution():
    cfg = ModelConfig(pretrained=False, fixed_size=[160, 128], box_score_thresh=0.0)
    model = build_detector(cfg, num_classes=3).eval()
    assert model.transform.fixed_size == (160, 128)
    with torch.no_grad():
        out = model([torch.rand(3, 300, 500)])[0]
    if len(out["boxes"]):
        assert out["boxes"][:, 2].max() <= 500 and out["boxes"][:, 3].max() <= 300
