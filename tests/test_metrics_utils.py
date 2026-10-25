import ast
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from lnrgst_mca import metrics_utils
from lnrgst_mca.metrics_utils import (
    calculate_recall,
    decompose_tensor,
    framewise_displacment_all_subjects_vectorized,
    mean_absolute_difference,
)

from conftest import make_affine, rotation_z

SRC = Path(__file__).resolve().parents[1] / "src"


def legacy_fd_all_subjects(translation_mca, angles_mca, r=50):
    """The loop-based FD_all_subjects removed in 58bbc58, kept here as the reference implementation."""
    n, n_mca, _ = translation_mca.shape
    out = np.zeros((n, n_mca))
    for i, j in product(range(n), range(n_mca)):
        d_rotation = (r * np.pi / 180) * np.sqrt(np.sum(angles_mca[i, j] ** 2))
        d_translation = np.sqrt(np.sum(translation_mca[i, j] ** 2))
        out[i, j] = d_rotation + d_translation
    return out


class TestDecomposeTensor:
    def test_identity(self):
        scales, translations, angles, shears = decompose_tensor(np.eye(4)[np.newaxis])
        np.testing.assert_allclose(scales, np.ones((1, 1, 3)))
        np.testing.assert_allclose(translations, 0, atol=1e-12)
        np.testing.assert_allclose(angles, 0, atol=1e-12)
        np.testing.assert_allclose(shears, 0, atol=1e-12)

    def test_known_affine_angles_in_degrees(self, known_affine):
        _, translations, angles, _ = decompose_tensor(known_affine[np.newaxis, np.newaxis])
        np.testing.assert_allclose(translations[0, 0], [1.0, -2.0, 3.0])
        np.testing.assert_allclose(angles[0, 0], [0.0, 0.0, 10.0], atol=1e-10)

    def test_output_shape_4d(self):
        tensor = np.tile(np.eye(4), (2, 5, 1, 1))
        for out in decompose_tensor(tensor):
            assert out.shape == (2, 5, 3)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            decompose_tensor(np.zeros((2, 3, 3)))


class TestFramewiseDisplacement:
    def test_matches_legacy_implementation(self):
        rng = np.random.default_rng(0)
        translations = rng.normal(size=(4, 10, 3))
        angles = rng.normal(scale=0.1, size=(4, 10, 3))
        np.testing.assert_allclose(
            framewise_displacment_all_subjects_vectorized(translations, angles),
            legacy_fd_all_subjects(translations, angles),
        )

    def test_pure_rotation_is_arc_length_on_50mm_sphere(self):
        angles = np.array([[[0.0, 0.0, 1.0]]])  # 1 degree
        fd = framewise_displacment_all_subjects_vectorized(np.zeros((1, 1, 3)), angles)
        np.testing.assert_allclose(fd, [[50 * np.pi / 180]])

    def test_zero_when_equal_to_reference(self):
        rng = np.random.default_rng(1)
        t = rng.normal(size=(3, 1, 3))
        a = rng.normal(size=(3, 1, 3))
        fd = framewise_displacment_all_subjects_vectorized(np.repeat(t, 5, axis=1), np.repeat(a, 5, axis=1), t, a)
        np.testing.assert_allclose(fd, 0, atol=1e-12)

    def test_radian_mode_equivalent(self):
        angles_deg = np.array([[[10.0, 0.0, 0.0]]])
        t = np.zeros((1, 1, 3))
        np.testing.assert_allclose(
            framewise_displacment_all_subjects_vectorized(t, angles_deg, mode="degree"),
            framewise_displacment_all_subjects_vectorized(t, np.radians(angles_deg), mode="radian"),
        )

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            framewise_displacment_all_subjects_vectorized(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), mode="grad")

    def test_end_to_end_from_matrices(self):
        # Two MCA runs: one identical to identity, one rotated 1 degree and shifted 1 mm.
        tensor = np.stack([np.eye(4), make_affine(rotation_z(1.0), (1.0, 0.0, 0.0))])[np.newaxis]
        _, translations, angles, _ = decompose_tensor(tensor)
        fd = framewise_displacment_all_subjects_vectorized(translations, angles)
        np.testing.assert_allclose(fd, [[0.0, 1.0 + 50 * np.pi / 180]], atol=1e-10)


class TestMeanAbsoluteDifference:
    @pytest.mark.parametrize("ieee", [np.array([1.0, 2.0]), np.array([[1.0], [2.0]])])
    def test_accepts_1d_and_2d_reference(self, ieee):
        mca = np.array([[1.0, 3.0], [2.0, 2.0]])
        np.testing.assert_allclose(mean_absolute_difference(mca, ieee), [1.0, 0.0])


def test_calculate_recall():
    assert calculate_recall([-1, -1, 1, 1]) == 0.5
    assert calculate_recall([]) == 0.0


def test_analysis_scripts_only_call_existing_metrics():
    """Guards against renames in metrics_utils silently breaking the analysis scripts (see FD_all_subjects)."""
    for script in SRC.glob("*.py"):
        tree = ast.parse(script.read_text())
        missing = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "metrics_utils"
            and not hasattr(metrics_utils, node.attr)
        }
        assert not missing, f"{script.name} references missing metrics_utils functions: {sorted(missing)}"
