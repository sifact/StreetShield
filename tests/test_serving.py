import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from streetshield.config import ModelConfig, load_config
from streetshield.inference import Predictor
from streetshield.modeling import build_detector
from streetshield.serving import BlurService, InvalidImageError, create_app, decode_image


@pytest.fixture(scope="module")
def client():
    cfg = load_config()
    model = build_detector(ModelConfig(pretrained=False, fixed_size=[160, 128]), num_classes=3).eval()
    predictor = Predictor(model, ["face", "plate"], cfg.inference, torch.device("cpu"))
    service = BlurService(predictor, cfg, {"epoch": 1, "arch": "test"})
    with TestClient(create_app(cfg, service)) as c:
        yield c


def _png(size=(200, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_model_info(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["backend"] == "torch:cpu"
    info = client.get("/api/model").json()
    assert info["classes"] == ["face", "plate"] and info["checkpoint_epoch"] == 1 and info["arch"] == "test"
    assert info["defaults"]["method"] in ("gaussian", "pixelate")


def test_blur_returns_image_and_echoes_params(client):
    res = client.post(
        "/api/blur", files={"file": ("a.png", _png(), "image/png")}, data={"score_thresh": "0.9", "method": "pixelate"}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert (body["width"], body["height"]) == (200, 120)
    assert body["params"]["score_thresh"] == 0.9 and body["params"]["method"] == "pixelate"
    assert set(body["counts"]) == {"face", "plate"}


def test_blur_rejects_bad_input(client):
    assert client.post("/api/blur", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 400
    bad = client.post("/api/blur", files={"file": ("a.png", _png(), "image/png")}, data={"score_thresh": "1.5"})
    assert bad.status_code == 422


def test_decode_image_downscales_large_uploads():
    image, original = decode_image(_png((3000, 1000)), max_side=1500)
    assert original == (3000, 1000) and max(image.size) == 1500
    with pytest.raises(InvalidImageError):
        decode_image(b"nope", 100)
