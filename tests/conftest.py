"""Shared fixtures: small affine files in each tool's on-disk format, built in tmp_path."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io


def rotation_z(degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_affine(rotation=np.eye(3), translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat


def write_flirt(path: Path, mat: np.ndarray) -> Path:
    """FLIRT writes a plain-text 4x4 matrix, using the .mat extension."""
    np.savetxt(path, mat)
    return path


def write_spm_octave(path: Path, mat: np.ndarray) -> Path:
    """run_affreg.m runs `save(path, 'M')` under Octave, which defaults to Octave's text format."""
    rows = "\n".join(" " + " ".join(f"{v:.17g}" for v in row) for row in mat)
    path.write_text(f"# Created by Octave\n# name: M\n# type: matrix\n# rows: 4\n# columns: 4\n{rows}\n")
    return path


def write_spm_matlab(path: Path, mat: np.ndarray) -> Path:
    """The same matrix saved as a real MATLAB file (e.g. `save -mat`)."""
    scipy.io.savemat(path, {"M": mat})
    return path


def write_ants(path: Path, mat: np.ndarray, fixed=(0.0, 0.0, 0.0)) -> Path:
    """ANTs *0GenericAffine.mat: MATLAB v4, 9 row-major linear terms + 3 translation terms, then the fixed center."""
    params = np.concatenate([mat[:3, :3].ravel(), mat[:3, 3]]).reshape(12, 1)
    scipy.io.savemat(
        path,
        {"AffineTransform_double_3_3": params, "fixed": np.asarray(fixed, dtype=float).reshape(3, 1)},
        format="4",
    )
    return path


@pytest.fixture
def known_affine() -> np.ndarray:
    return make_affine(rotation_z(10.0), translation=(1.0, -2.0, 3.0))


@pytest.fixture
def subjects_file(tmp_path) -> Path:
    path = tmp_path / "subjects.txt"
    path.write_text("sub-001\nsub-002\n")
    return path


@pytest.fixture
def registration_tree(tmp_path, known_affine):
    """
    A FLIRT-style output tree: <root>/ieee/<sub>.mat and <root>/mca/<i>/<sub>.mat for 3 MCA runs.
    sub-001 is complete; sub-002 has no IEEE file.
    """
    root = tmp_path / "flirt"
    (root / "ieee").mkdir(parents=True)
    write_flirt(root / "ieee" / "sub-001.mat", known_affine)
    for i in range(1, 4):
        run_dir = root / "mca" / str(i)
        run_dir.mkdir(parents=True)
        shifted = known_affine.copy()
        shifted[0, 3] += 0.01 * i
        write_flirt(run_dir / "sub-001.mat", shifted)
        write_flirt(run_dir / "sub-002.mat", known_affine)
    return root
