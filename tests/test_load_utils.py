import numpy as np
import pytest

from lnrgst_mca.load_utils import IEEE, MCA, _ensure_4x4, get_matrices, get_matrices_tensor, get_paths, load_file

from conftest import write_ants, write_flirt, write_spm_matlab, write_spm_octave


class TestEnsure4x4:
    def test_pads_3x4(self):
        mat = np.arange(12, dtype=float).reshape(3, 4)
        out = _ensure_4x4(mat)
        assert out.shape == (4, 4)
        np.testing.assert_array_equal(out[3], [0, 0, 0, 1])
        np.testing.assert_array_equal(out[:3], mat)

    def test_passes_4x4_through(self):
        mat = np.eye(4)
        assert _ensure_4x4(mat) is mat

    @pytest.mark.parametrize("shape", [(2, 4), (4, 3), (16,)])
    def test_rejects_other_shapes(self, shape):
        with pytest.raises(ValueError):
            _ensure_4x4(np.zeros(shape))


class TestLoadFile:
    @pytest.mark.parametrize(
        "writer", [write_flirt, write_spm_octave, write_spm_matlab, write_ants], ids=["flirt", "spm-octave", "spm-matlab", "ants"]
    )
    def test_round_trips_each_tool_format(self, tmp_path, known_affine, writer):
        path = writer(tmp_path / "affine.mat", known_affine)
        np.testing.assert_allclose(load_file(path), known_affine, atol=1e-12)

    def test_ants_ignores_fixed_center(self, tmp_path, known_affine):
        # Current behaviour: the ANTs center of rotation is not folded into the translation.
        path = write_ants(tmp_path / "affine.mat", known_affine, fixed=(10.0, 20.0, 30.0))
        np.testing.assert_allclose(load_file(path), known_affine, atol=1e-12)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "nope.mat")

    def test_garbage_file(self, tmp_path):
        path = tmp_path / "bad.mat"
        path.write_text("not a matrix\n")
        with pytest.raises(RuntimeError):
            load_file(path)


def test_get_paths_layout(tmp_path, subjects_file):
    paths = get_paths(tmp_path, subjects_file, n_mca=3, pattern="_ses-BL")

    assert list(paths) == ["sub-001", "sub-002"]
    assert paths["sub-001"][IEEE] == str(tmp_path / "ieee" / "sub-001_ses-BL.mat")
    assert paths["sub-001"][MCA] == [str(tmp_path / "mca" / str(i) / "sub-001_ses-BL.mat") for i in (1, 2, 3)]


def test_get_matrices_collects_errors(registration_tree, subjects_file):
    paths = get_paths(registration_tree, subjects_file, n_mca=3)
    matrices, errors = get_matrices(paths)

    assert list(matrices) == ["sub-001"]
    assert matrices["sub-001"][IEEE].shape == (4, 4)
    assert matrices["sub-001"][MCA].shape == (3, 4, 4)
    assert len(errors) == 1 and "sub-002" in errors[0] and "IEEE" in errors[0]


def test_get_matrices_keeps_partial_mca(registration_tree, subjects_file):
    (registration_tree / "mca" / "2" / "sub-001.mat").unlink()
    paths = get_paths(registration_tree, subjects_file, n_mca=3)
    matrices, errors = get_matrices(paths)

    assert matrices["sub-001"][MCA].shape == (2, 4, 4)
    assert any("sub-001" in e and "MCA" in e for e in errors)


def test_get_matrices_tensor_shapes(registration_tree, subjects_file):
    paths = get_paths(registration_tree, subjects_file, n_mca=3)
    mca, ieee, ids, _ = get_matrices_tensor(paths)

    assert mca.shape == (1, 3, 4, 4)
    assert ieee.shape == (1, 4, 4)
    assert ids == ["sub-001"]
