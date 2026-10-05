import numpy as np
import pytest

from blurring.inference import blur_regions


@pytest.fixture
def noisy_image():
    return np.random.default_rng(0).integers(0, 256, (100, 120, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", ["gaussian", "pixelate"])
def test_blur_changes_only_the_box(noisy_image, method):
    out = blur_regions(noisy_image, np.array([[20, 30, 60, 70]]), method)
    inside, original_inside = out[30:70, 20:60].astype(float), noisy_image[30:70, 20:60].astype(float)
    assert inside.std() < original_inside.std() * 0.5  # detail destroyed
    mask = np.ones(noisy_image.shape[:2], bool)
    mask[30:70, 20:60] = False
    np.testing.assert_array_equal(out[mask], noisy_image[mask])


def test_blur_does_not_modify_input_and_handles_edges(noisy_image):
    before = noisy_image.copy()
    blur_regions(noisy_image, np.array([[-10, -10, 5, 5], [110, 90, 200, 200], [5, 5, 5.5, 5.5]]))
    np.testing.assert_array_equal(noisy_image, before)


def test_unknown_method_raises(noisy_image):
    with pytest.raises(ValueError):
        blur_regions(noisy_image, np.zeros((0, 4)), "smudge")
